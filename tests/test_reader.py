"""
Tests for jmpio reader functionality
"""

import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import re
import sys

# Add parent directory to path so we can import our package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import read_jmp, scan_directory

# Directory with test data (set up by conftest.py)
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')


def test_read_jmp_existence():
    """Test that the read_jmp function exists"""
    assert callable(read_jmp)


def test_invalid_file():
    """Test reading a non-existent file"""
    with pytest.raises(FileNotFoundError):
        read_jmp("nonexistent_file.jmp")


def test_example1():
    """Test reading example1.jmp"""
    file_path = os.path.join(TEST_DATA_DIR, 'example1.jmp')
    
    # Skip if file doesn't exist (in case the test data wasn't properly set up)
    if not os.path.exists(file_path):
        pytest.skip(f"Test file {file_path} not found")
    
    df = read_jmp(file_path)
    
    # Test dimensions
    assert df.shape[0] == 4  # 4 rows
    assert df.shape[1] == 12  # 12 columns
    
    # Test column names
    expected_columns = [
        'ints', 'floats', 'charconstwidth', 'time', 'date', 'duration',
        'charconstwidth2', 'charvariable16', 'formula', 'pressures', 'char utf8',
        'charvariable8'
    ]
    assert set(df.columns) == set(expected_columns)
    
    # Test integer column
    assert df['ints'].tolist() == [1, 2, 3, 4]
    
    # Test float column
    assert df['floats'].tolist() == [11.1, 22.2, 33.3, 44.4]
    
    # Test constant width character column
    assert df['charconstwidth'].tolist() == ['a', 'b', 'c', 'd']
    
    # Test date/time columns
    assert pd.Timestamp(df['time'].iloc[0]).strftime('%Y-%m-%d %H:%M') == '1976-04-01 21:12'
    assert pd.Timestamp(df['date'].iloc[0]).strftime('%Y-%m-%d') == '2024-01-13'
    assert pd.isna(df['time'].iloc[3])
    assert pd.isna(df['date'].iloc[2])
    
    # Test duration
    assert isinstance(df['duration'].iloc[0], timedelta)
    assert df['duration'].iloc[0].total_seconds() == pytest.approx(2322, abs=2)
    
    # Test variable width character columns
    assert df['charvariable16'].iloc[0] == 'aa'
    assert df['charvariable16'].iloc[3].startswith('abcdefghijabcdefghij')
    
    # Test formula column
    assert df['formula'].tolist() == ['2', '4', '6', '8']
    
    # Test column with missing values
    assert df['pressures'].iloc[0] == pytest.approx(101.325)
    assert pd.isna(df['pressures'].iloc[1])
    
    # Test UTF-8 column
    # Note: exact UTF-8 characters may vary by platform, so just check types
    assert all(isinstance(s, str) for s in df['char utf8'])
    
    # Test variable width strings
    assert df['charvariable8'].tolist() == ['a', 'bb', 'cc', 'abcdefghijkl']


def test_compressed():
    """Test reading compressed.jmp"""
    file_path = os.path.join(TEST_DATA_DIR, 'compressed.jmp')
    
    # Skip if file doesn't exist
    if not os.path.exists(file_path):
        pytest.skip(f"Test file {file_path} not found")
    
    df = read_jmp(file_path)
    
    # Test numeric column - all values should be 1
    assert df['numeric'].fillna(0).eq(1).all()
    
    # Test character columns
    assert all(df['character1'] == 'a')
    assert all(df['character11'] == 'abcdefghijk')
    assert all(df['character130'] == 'abcdefghij' * 13)


def test_date_file():
    """Test reading date.jmp"""
    file_path = os.path.join(TEST_DATA_DIR, 'date.jmp')
    
    # Skip if file doesn't exist
    if not os.path.exists(file_path):
        pytest.skip(f"Test file {file_path} not found")
    
    df = read_jmp(file_path)
    
    # Test dates
    dates = pd.to_datetime(df['ddmmyyyy']).dt.strftime('%Y-%m-%d').tolist()
    expected_dates = ['2011-05-25', '1973-05-24', '2027-05-22', '2020-05-01']
    assert dates == expected_dates


def test_column_filtering():
    """Test column filtering features"""
    file_path = os.path.join(TEST_DATA_DIR, 'example1.jmp')
    
    # Skip if file doesn't exist
    if not os.path.exists(file_path):
        pytest.skip(f"Test file {file_path} not found")
    
    # Test selecting columns by name
    df = read_jmp(file_path, select=['ints', 'floats'])
    assert list(df.columns) == ['ints', 'floats']
    
    # Test selecting columns by index
    df = read_jmp(file_path, select=[0, 1])
    assert len(df.columns) == 2
    assert 'ints' in df.columns
    assert 'floats' in df.columns
    
    # Test selecting columns by regex
    df = read_jmp(file_path, select=[re.compile(r'^char')])
    for col in df.columns:
        assert col.startswith('char')
    
    # Test dropping columns
    df = read_jmp(file_path, drop=['ints', 'floats'])
    assert 'ints' not in df.columns
    assert 'floats' not in df.columns


def test_scan_directory():
    """Test scanning a directory for JMP files"""
    # This test requires the scan_directory function to be implemented
    result = scan_directory(TEST_DATA_DIR, recursive=False)
    
    # Check that we got a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check that we found at least some JMP files
    assert len(result) > 0
    
    # Check that all files have .jmp extension
    for file in result['filename']:
        assert file.lower().endswith('.jmp')


if __name__ == '__main__':
    pytest.main(['-v', __file__])