"""
Pipeline runners for running pipelines
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from mio.models import Pipeline


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


class SynchronousRunner(PipelineRunner):
    """
    Simple, synchronous pipeline runner.

    Just run the nodes in topological order and return from return nodes.
    """
