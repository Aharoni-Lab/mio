"""
Pipeline runners for running pipelines
"""

from abc import ABC, abstractmethod
from collections.abc import Generator, MutableSequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count
from logging import Logger
from typing import TYPE_CHECKING, Any, Optional, Self

from mio import init_logger
from mio.models import Pipeline
from mio.models.pipeline import Edge, Event, Node, Source

if TYPE_CHECKING:
    from mio.sinks import Return


@dataclass
class EventStore:
    """
    Container class for storing and retrieving events by node and slot
    """

    events: MutableSequence = field(default_factory=list)
    counter: count = field(default_factory=count)

    def add(self, values: dict[str, Any], node_id: str) -> None:
        """
        Add the result of a :meth:`.Node.process` call to the event store.

        Split the dictionary of values into separate :class:`.Event` s,
        store along with current timestamp

        Args:
            values (dict): Dict emitted by a :meth:`.Node.process` call
            node_id (str): ID of the node that emitted the events
        """
        if values is None:
            return
        timestamp = datetime.now()
        for slot, value in values.items():
            self.events.append(
                Event(
                    id=next(self.counter),
                    timestamp=timestamp,
                    node_id=node_id,
                    slot=slot,
                    value=value,
                )
            )

    def get(self, node_id: str, slot: str) -> Optional[Event]:
        """
        Get the event with the matching node_id and slot name

        Returns the most recent matching event, as for now we assume that
        each combination of `node_id` and `slot` is emitted only once per processing cycle,
        and we assume processing cycles are independent (and thus our events are cleared)

        ``None`` in the case that the event has not been emitted
        """
        event = [e for e in self.events if e["node_id"] == node_id and e["slot"] == slot]
        return None if len(event) == 0 else event[-1]

    def gather(self, edges: list[Edge]) -> Optional[dict]:
        """
        Gather events into a form that can be consumed by a :meth:`.Node.process` method,
        given the collection of inbound edges (usually from :meth:`.Pipeline.in_edges` ).

        If none of the requested events have been emitted, return ``None``.

        If all of the requested events have been emitted, return a kwarg-like dict

        If some of the requested events are missing but others are present,
        return ``None`` for any missing events.

        .. todo::

            Add an example

        """
        ret = {}
        for edge in edges:
            event = self.get(edge.source_node.id, edge.source_slot)
            value = None if event is None else event["value"]
            ret[edge.target_slot] = value

        return None if not ret or all(val is None for val in ret.values()) else ret

    def clear(self) -> None:
        """
        Clear events for this round of processing.

        Does not reset the counter (to continue giving unique ids to the next round's events)
        """
        self.events = []


@dataclass
class PipelineRunner(ABC):
    """
    Abstract parent class for pipeline runners.

    Pipeline runners handle calling the nodes and passing the
    events returned by them to each other. Each runner may do so
    however it needs to (synchronously, asynchronously, alone or as part of a cluster, etc.)
    as long as it satisfies this abstract interface.
    """

    pipeline: Pipeline
    store: EventStore = field(default_factory=EventStore)

    _logger: Logger = field(default_factory=lambda: init_logger("pipeline.runner"))

    @abstractmethod
    def process(self) -> Optional[dict[str, Any]]:
        """
        Process one step of data from each of the sources,
        passing intermediate data to any subscribed nodes in a chain.

        The `process` method normally does not return anything,
        except when using the special :class:`.ReturnSink` node -
        if there are :class:`.ReturnSink` nodes in a :class:`.Pipeline` graph,
        then each call to `process` will return a dictionary with one key
        (from the :class:`.ReturnSink`'s `key` config value) and one value for each
        :class:`.ReturnSink`.
        """

    @abstractmethod
    def start(self) -> None:
        """
        Start processing data with the pipeline graph
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Stop processing data with the pipeline graph
        """

    def gather_input(self, node: Node) -> Optional[dict[str, Any]]:
        """
        Gather input to give to the passed Node from the :attr:`.PipelineRunner.store`

        Returns:
            dict: kwargs to pass to :meth:`.Node.process` if matching events are present
            dict: empty dict if Node is a :class:`.Source`
            None: if no input is available
        """
        if isinstance(node, Source):
            return {}

        edges = self.pipeline.in_edges(node)
        return self.store.gather(edges)

    def gather_return(self) -> Optional[dict]:
        """
        If any :class:`.Return` nodes are in the pipeline,
        gather their return values to return from :meth:`.PipelineRunner.process`

        Returns:
            dict: of the Return sink's key mapped to the returned value,
            None: if there are no :class:`.Return` sinks in the pipeline
        """
        ret = {}
        for sink in self.pipeline.sinks.values():
            if sink.name != "return":
                continue
            sink: Return
            val = sink.get(keep=False)
            ret.update(val)

        if not ret:
            return None
        else:
            return ret


class SynchronousRunner(PipelineRunner):
    """
    Simple, synchronous pipeline runner.

    Just run the nodes in topological order and return from return nodes.
    """

    @contextmanager
    def start(self) -> Generator[Self, None, None]:
        """
        Start processing data with the pipeline graph.

        Returns a contextmanager that should be used like this:

        .. code-block:: python

            with sync_runner.start() as runner:
                output = runner.process()
                # do something...

        """
        # TODO: lock for re-entry
        try:
            for node in self.pipeline.nodes.values():
                node.start()
            yield self
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop all nodes processing"""
        # TODO: lock to ensure we've been started
        for node in self.pipeline.nodes.values():
            node.stop()

    def process(self) -> Optional[dict[str, Any]]:
        """
        Iterate through nodes in topological order,
        calling their process method and passing events as they are emitted.
        """
        self.store.clear()

        graph = self.pipeline.graph()
        graph.prepare()

        while graph.is_active():
            for node_id in graph.get_ready():
                node = self.pipeline.nodes[node_id]
                node_input = self.gather_input(node)
                if node_input is None:
                    graph.done(node_id)
                    self._logger.debug(f"Node {node_id} received no input, skipping")
                    continue
                value = node.process(**node_input)
                self.store.add(value, node_id)
                graph.done(node_id)
                self._logger.debug(f"Node {node_id} emitted %s", value)

        return self.gather_return()
