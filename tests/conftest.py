"""
Pytest configuration
"""

import os
import shutil
import pytest
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Directory for test data
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')

# Source directory for JMP test files
ORIG_TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'JMPReader.jl', 'test'))


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """
    Copy test files from the original Julia package before running tests
    and clean up after tests are finished
    """
    # Create test data directory
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Check if source directory exists
    if not os.path.exists(ORIG_TEST_DIR):
        pytest.skip(f"Original test directory {ORIG_TEST_DIR} not found")
    
    # Copy test JMP files
    for file in os.listdir(ORIG_TEST_DIR):
        if file.endswith('.jmp'):
            src_file = os.path.join(ORIG_TEST_DIR, file)
            dst_file = os.path.join(TEST_DATA_DIR, file)
            shutil.copy2(src_file, dst_file)
    
    yield
    
    # Clean up
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)