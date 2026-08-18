"""Pytest configuration and fixtures for search_names tests."""

import multiprocessing
import sys
from contextlib import suppress

import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_multiprocessing():
    """Ensure multiprocessing resources are properly cleaned up after tests."""
    yield

    # Force cleanup of any remaining multiprocessing resources
    # This prevents hanging after test completion
    if hasattr(multiprocessing, "active_children"):
        for child in multiprocessing.active_children():
            child.terminate()
            child.join(timeout=1)

    # Clean up any managers
    if hasattr(multiprocessing, "_resource_tracker"):
        multiprocessing._resource_tracker._stop()


@pytest.fixture(autouse=True)
def reset_multiprocessing_state():
    """Reset multiprocessing state between tests."""
    yield

    # Ensure no lingering processes between tests
    if hasattr(multiprocessing, "active_children"):
        for child in multiprocessing.active_children():
            if child.is_alive():
                child.terminate()
                child.join(timeout=0.5)


def pytest_configure(config):
    """Configure pytest settings."""
    # Set multiprocessing start method to 'spawn' for consistency
    # This helps avoid issues with forking and module imports
    if sys.platform != "win32":
        with suppress(RuntimeError):
            multiprocessing.set_start_method("spawn", force=True)
