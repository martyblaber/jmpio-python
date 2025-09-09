"""
Tests for utility functions
"""

import io
import struct

import numpy as np
import pandas as pd
import pytest

from jmpio.constants import JMP_STARTDATE, MAGIC_JMP
from jmpio.utils import (
    bit_cat,
    check_magic,
    hex_to_rgb,
    read_reals,
    read_string,
    sentinel_to_missing,
    to_datetime,
)


class TestReadString:
    def test_read_string_width1(self):
        """Test reading string with width 1"""
        # Create a test binary file in memory
        binary_data = struct.pack("b", 5) + b"hello"
        file = io.BytesIO(binary_data)

        result = read_string(file, 1)
        assert result == "hello"

    def test_read_string_width2(self):
        """Test reading string with width 2"""
        binary_data = struct.pack("h", 5) + b"hello"
        file = io.BytesIO(binary_data)

        result = read_string(file, 2)
        assert result == "hello"

    def test_read_string_width4(self):
        """Test reading string with width 4"""
        binary_data = struct.pack("i", 5) + b"hello"
        file = io.BytesIO(binary_data)

        result = read_string(file, 4)
        assert result == "hello"

    def test_read_string_invalid_width(self):
        """Test reading string with invalid width"""
        file = io.BytesIO(b"")

        with pytest.raises(ValueError, match="Invalid string width 3"):
            read_string(file, 3)


class TestReadReals:
    def test_read_reals_float64(self):
        """Test reading float64 values"""
        values = [1.1, 2.2, 3.3, 4.4]
        binary_data = np.array(values, dtype=np.float64).tobytes()
        file = io.BytesIO(binary_data)

        result = read_reals(file, np.float64, 4)
        assert np.allclose(result, values)

    def test_read_reals_int32(self):
        """Test reading int32 values"""
        values = [1, 2, 3, 4]
        binary_data = np.array(values, dtype=np.int32).tobytes()
        file = io.BytesIO(binary_data)

        result = read_reals(file, np.int32, 4)
        assert np.array_equal(result, values)


class TestToDatetime:
    def test_to_datetime(self):
        """Test converting JMP timestamps to datetime"""
        # Create a timestamp array (seconds since JMP epoch)
        seconds_since_epoch = np.array([0.0, 86400.0])  # 0 and 1 day after JMP epoch

        result = to_datetime(seconds_since_epoch)

        # Expected results
        expected_timestamps = np.array(
            [
                np.datetime64(JMP_STARTDATE),
                np.datetime64(JMP_STARTDATE) + np.timedelta64(1, "D"),
            ]
        )

        assert np.array_equal(result, expected_timestamps)

    def test_to_datetime_with_nan(self):
        """Test handling NaN values in timestamps"""
        timestamps = np.array([0.0, np.nan, 86400.0])

        result = to_datetime(timestamps)

        # Check that the NaN is converted to NaT
        assert np.isnat(result[1])
        assert not np.isnat(result[0])
        assert not np.isnat(result[2])


class TestCheckMagic:
    def test_check_magic_valid(self):
        """Test checking magic bytes with valid file"""
        file = io.BytesIO(MAGIC_JMP + b"extra data")

        assert check_magic(file) is True
        # Position should be reset
        assert file.tell() == 0

    def test_check_magic_invalid(self):
        """Test checking magic bytes with invalid file"""
        file = io.BytesIO(b"invalid magic bytes")

        assert check_magic(file) is False
        # Position should be reset
        assert file.tell() == 0


class TestSentinelToMissing:
    def test_sentinel_to_missing_float(self):
        """Test converting NaN to missing in floats"""
        data = np.array([1.0, np.nan, 3.0], dtype=np.float64)

        result = sentinel_to_missing(data)

        # Should be a pandas Series with NA
        assert isinstance(result, pd.Series)
        assert result.isna()[1]
        assert not result.isna()[0]
        assert not result.isna()[2]

    def test_sentinel_to_missing_int8(self):
        """Test converting sentinel values to missing in int8"""
        sentinel = np.iinfo(np.int8).min + 1
        data = np.array([1, sentinel, 3], dtype=np.int8)

        result = sentinel_to_missing(data)

        # Should convert sentinel to NA and use pandas Int8 type
        assert isinstance(result, pd.Series)
        assert result.isna()[1]
        assert not result.isna()[0]
        assert not result.isna()[2]
        assert result.dtype == "Int8"


class TestHexToRgb:
    def test_hex_to_rgb(self):
        """Test converting hex colors to RGB values"""
        assert hex_to_rgb("#000000") == (0.0, 0.0, 0.0)
        assert hex_to_rgb("#FF0000") == (1.0, 0.0, 0.0)
        assert hex_to_rgb("#00FF00") == (0.0, 1.0, 0.0)
        assert hex_to_rgb("#0000FF") == (0.0, 0.0, 1.0)
        assert hex_to_rgb("#FFFFFF") == (1.0, 1.0, 1.0)

        # Test with hash prefix
        assert hex_to_rgb("#80A0C0") == (
            0.5019607843137255,
            0.6274509803921569,
            0.7529411764705882,
        )

        # Test without hash prefix
        assert hex_to_rgb("80A0C0") == (
            0.5019607843137255,
            0.6274509803921569,
            0.7529411764705882,
        )


class TestBitCat:
    def test_bit_cat(self):
        """Test concatenating two bytes into a 16-bit integer"""
        assert bit_cat(0x12, 0x34) == 0x1234
        assert bit_cat(0xFF, 0xFF) == 0xFFFF
        assert bit_cat(0x00, 0x00) == 0x0000
        assert bit_cat(0x00, 0xFF) == 0x00FF
        assert bit_cat(0xFF, 0x00) == 0xFF00
