"""
Base ABCs for pipeline classes
"""

import sys
from abc import abstractmethod
from typing import ClassVar, Final, Generic, Optional, TypedDict, TypeVar, Union, final

from pydantic import Field, field_validator, model_validator

from mio.exceptions import ConfigurationMismatchError
from mio.models.models import MiniscopeConfig, PipelineModel

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

T = TypeVar("T")
"""
Input Type typevar
"""
U = TypeVar("U")
"""
Output Type typevar
"""


class _NodeMap(TypedDict):
    source: str
    target: str


class NodeConfig(TypedDict):
    """
    Abstract parent TypedDict that each node inherits from to define
    what fields it needs to be configured.
    """


class NodeSpecification(MiniscopeConfig):
    """
    Specification for a single processing node within a pipeline .yaml file.
    Distinct from a :class:`.NodeConfig`, which is a generic TypedDict that each
    node defines to declare its parameterization.
    """

    type_: str = Field(..., alias="type")
    """
    Shortname of the type of node this configuration is for.
    
    Subclasses should override this with a default.
    """
    id: str
    """The unique identifier of the node"""
    outputs: Optional[list[_NodeMap]] = None
    """List of Node IDs to be used as output"""
    config: Optional[NodeConfig] = None
    """Additional configuration for this node, parameterized by a TypedDict for the class"""
    passed: Optional[list[str]] = None
    """
    List of config values that must be passed when the pipeline is instantiated
    """
    fill: Optional[dict[str, str]] = None
    """
    Values in the node config that should be dynamically filled from other nodes in the pipeline.
    
    Specified as {node_id}.{attribute}, these specify attributes and properties
    on the instantiated node class, not the config values for that node.
    
    This is useful for accessing some properties that might not be known until runtime
    like width and height of an input image.
    
    Examples:
    
        For a node class `camera` that has property `frame_width`,
        and node class `process` that has config value `width`,
        we would fill the config value like this: 
        
        .. code-block:: yaml
        
            nodes:
              cam:
                type: camera
              proc:
                type: process
                fill:
                  width: cam.frame_width
                  
        The Pipeline class will then do something like this on instantiation:
        
        .. code-block:: python
        
            pipeline = PipelineConfig(**the_above_values)
            
            cam = CameraNode(config=pipeline.nodes['cam'].config)
            
            proc_config = pipeline.nodes['proc'].config
            proc_config['width'] = cam.frame_width
            proc = ProcessingNode(config=proc_config)    
    
    """


class PipelineConfig(MiniscopeConfig):
    """
    Configuration for the nodes within a pipeline
    """

    required_nodes: ClassVar[Optional[dict[str, str]]] = None
    """
    id: type mapping that a subclass can use to require a set of node types 
    with specific IDs be present
    """

    nodes: dict[str, NodeSpecification] = Field(default_factory=dict)
    """The nodes that this pipeline configures"""

    @model_validator(mode="after")
    def validate_required_nodes(self) -> Self:
        """Ensure required nodes are present, if any"""
        if self.required_nodes is not None:
            for id_, type_ in self.required_nodes.items():
                assert id_ in self.nodes, f"Node ID {id_} not in {self.nodes.keys()}"
                assert self.nodes[id_].type_ == type_, f"Node ID {id_} is not of type {type_}"
        return self

    @field_validator("nodes", mode="before")
    @classmethod
    def fill_node_ids(cls, value: dict[str, dict]) -> dict[str, dict]:
        """
        Roll down the `id` from the key in the `nodes` dictionary into the node config
        """
        assert isinstance(value, dict)
        for id, node in value.items():
            if "id" not in node:
                node["id"] = id
        return value


class Node(PipelineModel, Generic[T, U]):
    """A node within a processing pipeline"""

    name: ClassVar[str]
    """
    Shortname for this type of node to match configs to node types
    """

    id: str
    """Unique identifier of the node"""
    config: Optional[NodeConfig] = None

    input_type: ClassVar[type[T]]
    inputs: dict[str, Union["Source", "Transform"]] = Field(default_factory=dict)
    output_type: ClassVar[type[U]]
    outputs: dict[str, Union["Sink", "Transform"]] = Field(default_factory=dict)

    def start(self) -> None:
        """
        Start producing, processing, or receiving data.

        Default is a no-op.
        Subclasses do not need to override if they have no initialization logic.
        """
        pass

    def stop(self) -> None:
        """
        Stop producing, processing, or receiving data

        Default is a no-op.
        Subclasses do not need to override if they have no deinit logic.
        """
        pass

    @classmethod
    def from_config(cls, config: NodeSpecification) -> Self:
        """
        Create a node from its config
        """
        return cls(id=config.id, config=config)

    @classmethod
    @final
    def node_types(cls) -> dict[str, type["Node"]]:
        """
        Map of all imported :attr:`.Node.name` names to node classes
        """
        node_types = {}
        to_check = cls.__subclasses__()
        while to_check:
            node = to_check.pop()
            if node.name in node_types:
                raise ValueError(
                    f"Repeated node name identifier: {node.name}, found in:\n"
                    f"- {node_types[node.name]}\n- {node}"
                )
            node_types[node.name] = node
            to_check.extend(node.__subclasses__())
        return node_types


class Source(Node, Generic[T, U]):
    """A source of data in a processing pipeline"""

    inputs: Final[None] = None
    input_type: ClassVar[None] = None

    @abstractmethod
    def process(self) -> U:
        """
        Process some data, returning an output.


        .. note::

            The `process` method should not directly call or pass
            data to subscribed output nodes, but instead return the output
            and allow a containing pipeline class to handle dispatching data.

        """


class Sink(Node, Generic[T, U]):
    """A sink of data in a processing pipeline"""

    output_type: ClassVar[None] = None
    outputs: Final[None] = None

    @abstractmethod
    def process(self, data: T) -> None:
        """
        Process some incoming data, returning None

        .. note::

            The `process` method should not directly be called or passed data,
            but instead should be called by a containing pipeline class.

        """


class Transform(Node, Generic[T, U]):
    """
    An intermediate processing node that transforms some input to output
    """

    @abstractmethod
    def process(self, data: T) -> U:
        """
        Process some incoming data, yielding a transformed output

        .. note::

            The `process` method should not directly call or be called by
            output or input nodes, but instead return the output
            and allow a containing pipeline class to handle dispatching data.

        """


class Pipeline(PipelineModel):
    """
    A graph of nodes transforming some input source(s) to some output sink(s)

    The Pipeline model is a container for a set of nodes that are fully instantiated
    (e.g. have their "passed" and "fill" keys processed) and connected.
    It does not handle running the pipeline -- that is handled by a PipelineRunner.
    """

    nodes: dict[str, Node] = Field(default_factory=dict)
    """
    Dictionary mapping all nodes from their ID to the instantiated node.
    """

    @property
    def sources(self) -> dict[str, "Source"]:
        """All :class:`.Source` nodes in the processing graph"""
        return {k: v for k, v in self.nodes.items() if isinstance(v, Source)}

    @property
    def transforms(self) -> dict[str, "Transform"]:
        """All :class:`.Transform` s in the processing graph"""
        return {k: v for k, v in self.nodes.items() if isinstance(v, Transform)}

    @property
    def sinks(self) -> dict[str, "Sink"]:
        """All :class:`.Sink` nodes in the processing graph"""
        return {k: v for k, v in self.nodes.items() if isinstance(v, Sink)}

    @classmethod
    def from_config(cls, config: PipelineConfig) -> Self:
        """
        Instantiate a pipeline model from its configuration
        """
        types = Node.node_types()

        nodes = {k: types[v.type_].from_config(v) for k, v in config.nodes.items()}
        nodes = connect_nodes(nodes)
        return cls(nodes=nodes)


def connect_nodes(nodes: dict[str, Node]) -> dict[str, Node]:
    """
    Provide references to instantiated nodes
    """

    for node in nodes.values():
        if node.config.inputs and node.inputs is None:
            raise ConfigurationMismatchError(
                "inputs found in node configuration, but node type allows no inputs!\n"
                f"node: {node.model_dump()}"
            )
        if node.config.outputs and not hasattr(node, "outputs"):
            raise ConfigurationMismatchError(
                "outputs found in node configuration, but node type allows no outputs!\n"
                f"node: {node.model_dump()}"
            )

        node.inputs.update({id: nodes[id] for id in node.config.inputs})
        node.outputs.update({id: nodes[id] for id in node.config.outputs})
    return nodes
