"""
Custom exceptions!
"""


class ConfigurationError(ValueError):
    """Base exception class for errors in configuration"""


class InvalidSDException(Exception):
    """
    Raised when :class:`.io.WireFreeMiniscope` is used with a drive that doesn't have the
    appropriate WRITE KEYS in its header
    """


class EndOfRecordingException(StopIteration):
    """
    Raised when :class:`.io.WireFreeMiniscope` is at the end of the available recording!
    """


class SDException(Exception):
    """
    Base class for SDcard-specific errors
    """


class ReadHeaderException(SDException, RuntimeError):
    """
    Raised when a given frame's header cannot be read!
    """


class StreamError(RuntimeError):
    """
    Base class for errors while streaming data
    """


class StreamReadError(StreamError):
    """
    Error while reading streaming data from a device
    """


class DeviceError(RuntimeError):
    """
    Base class for errors when communicating with or configuring sources
    """


class DeviceOpenError(DeviceError):
    """
    Error opening a connection to a device
    """


class DeviceConfigurationError(DeviceError, ConfigurationError):
    """
    Error while configuring a device
    """


class ConfigurationMismatchError(ConfigurationError):
    """
    Mismatch between the fields in some config model and the fields in the model it is configuring
    """

class PipelineError(Exception):
    """
    Base exception type for pipeline errors
    """

class PipelineRunningError(PipelineError, RuntimeError):
    """
    A pipeline has been requested to start, but it is already started!
    """