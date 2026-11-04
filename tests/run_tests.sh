#!/usr/bin/env bash
# ================================================================================================
# run_tests.sh-Comprehensive Test Runner for the Carbon-Aware Deep Reinforcement Learning Platform
# ================================================================================================
# This script discovers and runs all tests in the /tests/ directory using pytest.
# It includes environment checks, optional coverage reporting, and proper 
# exit-code propagation for CI/CD pipelines.
#
# Usage:
#   ./scripts/run_tests.sh                  # Run all tests
#   ./scripts/run_tests.sh --coverage       # Run tests with coverage report
#   ./scripts/run_tests.sh tests/test_agent.py # Run a specific test file
#   ./scripts/run_tests.sh -h               # Show help
# =============================================================================

set -o pipefail

# -----------------------------------------------------------------------------
# Configuration & Colors
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$PROJECT_ROOT/tests"

# ANSI Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE} $1 ${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

show_help() {
    echo "Usage: $0 [OPTIONS] [pytest_args...]"
    echo ""
    echo "Options:"
    echo "  -h, --help       Show this help message and exit"
    echo "  -c, --coverage   Generate a coverage report (requires pytest-cov)"
    echo "  -q, --quick      Run tests without verbose output"
    echo ""
    echo "Examples:"
    echo "  $0                       # Run all tests verbosely"
    echo "  $0 --coverage            # Run all tests with coverage"
    echo "  $0 tests/test_agent.py   # Run a specific test file"
    echo "  $0 -k 'test_forward'     # Run tests matching a keyword"
}

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
COVERAGE=false
VERBOSE="-v"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) show_help; exit 0 ;;
        -c|--coverage) COVERAGE=true; shift ;;
        -q|--quick) VERBOSE=""; shift ;;
        *) break ;; # Stop parsing options, pass the rest to pytest
    esac
done

# -----------------------------------------------------------------------------
# Pre-flight Checks
# -----------------------------------------------------------------------------
print_header "Carbon-Aware Swarm Platform — Test Runner"

# 1. Check if test directory exists
if [ ! -d "$TEST_DIR" ]; then
    print_error "Test directory not found at $TEST_DIR"
    exit 1
fi

# 2. Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_error "pytest is not installed or not in PATH."
    echo "Please install it using: pip install -r requirements-dev.txt"
    exit 1
fi

# 3. Check if virtual environment is active (Optional but recommended warning)
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}WARNING: No active virtual environment detected.${NC}"
    echo -e "${YELLOW}It is recommended to run tests inside your project's venv.${NC}\n"
fi

# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
echo "Project Root : $PROJECT_ROOT"
echo "Test Directory: $TEST_DIR"
echo "Coverage     : $COVERAGE"
echo ""

# Build the pytest command
CMD="pytest $TEST_DIR $VERBOSE --tb=short"

if [ "$COVERAGE" = true ]; then
    # Check if pytest-cov is installed
    if python -c "import pytest_cov" &> /dev/null; then
        CMD="$CMD --cov=src --cov-report=term-missing --cov-report=html:htmlcov"
    else
        echo -e "${YELLOW}WARNING: pytest-cov is not installed. Skipping coverage.${NC}"
        echo "Install it with: pip install pytest-cov"
    fi
fi

# Append any extra arguments passed to the script (e.g., specific file or -k keyword)
CMD="$CMD $@"

echo -e "${BLUE}Running:${NC} $CMD"
echo ""

# Run pytest and capture the exit code
eval "$CMD"
EXIT_CODE=$?

# -----------------------------------------------------------------------------
# Post-Execution Summary
# -----------------------------------------------------------------------------
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed successfully!${NC}"
elif [ $EXIT_CODE -eq 1 ]; then
    echo -e "${RED}❌ Some tests failed.${NC}"
elif [ $EXIT_CODE -eq 2 ]; then
    echo -e "${RED}⚠️  Test execution was interrupted by the user.${NC}"
elif [ $EXIT_CODE -eq 3 ]; then
    echo -e "${RED}⚠️  Internal pytest error.${NC}"
elif [ $EXIT_CODE -eq 4 ]; then
    echo -e "${RED}⚠️  pytest command line usage error.${NC}"
elif [ $EXIT_CODE -eq 5 ]; then
    echo -e "${YELLOW}⚠️  No tests were collected.${NC}"
fi

# Open HTML coverage report if generated
if [ "$COVERAGE" = true ] && [ $EXIT_CODE -eq 0 ]; then
    if [ -f "htmlcov/index.html" ]; then
        echo -e "\n${BLUE}Coverage report generated at: htmlcov/index.html${NC}"
        # Uncomment the line below to auto-open in browser (macOS/Linux)
        # open htmlcov/index.html || xdg-open htmlcov/index.html
    fi
fi

# Propagate the exit code to the shell/CI pipeline
exit $EXIT_CODE