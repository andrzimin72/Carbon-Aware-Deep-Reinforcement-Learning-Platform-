"""
Test package for the Carbon-Aware Deep Reinforcement Learning Platform.

This package contains pytest-based unit and integration tests for all
components of the platform. Tests are organized by module:

    tests/
    ├── conftest.py                    # Shared fixtures
    ├── test_watttime_client.py        # WattTime API client tests
    ├── test_pipeline.py               # Data pipeline tests
    ├── test_environment.py            # Gym environment tests
    └── test_agent.py                  # PPO agent tests

Running tests:
    # Run all tests
    pytest tests/ -v

    # Run with coverage
    pytest tests/ --cov=src --cov-report=html

    # Run specific test file
    pytest tests/test_watttime_client.py -v

    # Run specific test class
    pytest tests/test_watttime_client.py::TestAuthentication -v

    # Run specific test
    pytest tests/test_watttime_client.py::TestAuthentication::test_login_success -v

Test fixtures:
    Shared fixtures are defined in conftest.py and automatically available
    to all test modules. Key fixtures include:
        - mock_watttime_*: Mock API responses
        - temp_cache_dir: Temporary cache directory
        - sample_replay_data: Synthetic replay buffer data
        - sample_env: Pre-configured Gym environment
        - sample_policy: Pre-configured PPO policy

Note:
    This is a test package, not a library package. It should not be imported
    by production code. The __init__.py file exists primarily to make pytest
    discovery work correctly and to document the test structure.
"""

# This file intentionally left mostly empty.
# Test discovery is handled by pytest, not by imports.

__version__ = "1.0.0"
