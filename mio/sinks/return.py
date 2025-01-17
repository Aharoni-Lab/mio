"""
Special Return sink that pipeline runners use to return values from :meth:`.PipelineRunner.process`
"""

from typing import Any, TypedDict

from mio.models.pipeline import Sink, T


class ReturnConfig(TypedDict):
    """
    Config for return nodes
    """

    key: str
    """The key to use in the returned dictionary"""


class Return(Sink):
    """
    Special sink node that returns values from a pipeline runner's `process` method
    """

    name = "return"
    input_type = Any

    config: ReturnConfig

    _value: Any = None

    def process(self, data: T) -> None:
        """
        Store the incoming value to retrieve later with :meth:`.get`
        """
        self._value = data

    def get(self) -> dict[str, T]:
        """
        Get the stored value from the process call
        """
        return {self.config["key"]: self._value}
