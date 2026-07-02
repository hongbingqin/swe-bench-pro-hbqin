#!/bin/bash
set -e

# --color=no + cleared addopts: the repo pins `addopts = "--color=yes"`, which
# injects ANSI codes around PASSED/FAILED and breaks parser.py's regex.
PYTEST_OPTS="-v --tb=short --color=no -o addopts= -p no:cacheprovider -p no:xdist"

run_all_tests() {
  echo "Running all tests..."
  cd /app
  python -m pytest $PYTEST_OPTS || true
}

run_selected_tests() {
  local test_files=("$@")
  echo "Running selected tests: ${test_files[@]}"
  cd /app

  for test_file in "${test_files[@]}"; do
    echo "Running test: $test_file"
    test_path=$(echo "$test_file" | sed 's/::.*//')
    python -m pytest "$test_path" $PYTEST_OPTS || true
  done
}

if [ $# -eq 0 ]; then
  run_all_tests
  exit $?
fi

if [[ "$1" == *","* ]]; then
  IFS=',' read -r -a TEST_FILES <<< "$1"
else
  TEST_FILES=("$@")
fi

run_selected_tests "${TEST_FILES[@]}"
