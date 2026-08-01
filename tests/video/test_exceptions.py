"""Unit tests for video exception hierarchy."""

import pytest
from app.video.exceptions import (
    AugmentationError,
    ComponentNotFoundError,
    ConfigurationError,
    DatasetError,
    DuplicateRegistrationError,
    ModelError,
    PipelineError,
    PreprocessingError,
    RegistryError,
    TrainingError,
    VideoException,
)


def test_video_exception_hierarchy():
    assert issubclass(ConfigurationError, VideoException)
    assert issubclass(DatasetError, VideoException)
    assert issubclass(PreprocessingError, VideoException)
    assert issubclass(AugmentationError, VideoException)
    assert issubclass(ModelError, VideoException)
    assert issubclass(TrainingError, VideoException)
    assert issubclass(PipelineError, VideoException)
    assert issubclass(RegistryError, VideoException)
    assert issubclass(ComponentNotFoundError, RegistryError)
    assert issubclass(DuplicateRegistrationError, RegistryError)


def test_raise_video_exceptions():
    with pytest.raises(VideoException):
        raise ConfigurationError("Config invalid")

    with pytest.raises(RegistryError):
        raise ComponentNotFoundError("Component missing")

    with pytest.raises(RegistryError):
        raise DuplicateRegistrationError("Already registered")
