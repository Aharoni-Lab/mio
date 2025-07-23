"""
Shared CLI utils
"""

from os import PathLike
from pathlib import Path
from typing import Optional, Union

from click import Context, Parameter, ParamType


class ConfigIDOrPath(ParamType):
    """
    A custom click type to accept either a config `id` or a path
    as input, resolving relative paths first against
    the current working directory and second against the user config directory.
    """

    name = "config-id-or-path"

    def convert(
        self, value: Union[str, PathLike[str]], param: Optional[Parameter], ctx: Optional[Context]
    ) -> Union[str, Path]:
        """
        If something looks like a yaml file, return as a path, otherwise return unchanged.

        Don't do validation here, the Config model will handle that on instantiation.
        """
        if value.endswith(".yaml") or value.endswith(".yml"):
            value = Path(value)
        return value
