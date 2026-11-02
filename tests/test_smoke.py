"""Basic package import smoke test."""

from study_planner import __version__


def test_package_is_importable() -> None:
    """The package exposes its initial version."""
    assert __version__ == "0.1.0"
