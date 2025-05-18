"""
Utility functions for reading and writing JMP files
"""

import struct
import re
import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta
from typing import List, Optional, Union, Any, TypeVar, BinaryIO, Tuple, cast
import mmap
from .constants import JMP_STARTDATE, MAGIC_JMP

T = TypeVar('T')


def read_string(file: BinaryIO, width: int) -> str:
    """
    Read a string from a binary file with specified width encoding
    
    Parameters:
    -----------
    file : BinaryIO
        Open file handle
    width : int
        Width of the string length encoding (1, 2, or 4 bytes)
        
    Returns:
    --------
    str
        The decoded string
    """
    if width not in [1, 2, 4]:
        raise ValueError(f"Invalid string width {width}")
    
    if width == 1:
        length = struct.unpack('b', file.read(1))[0]
    elif width == 2:
        length = struct.unpack('h', file.read(2))[0]
    else:  # width == 4
        length = struct.unpack('i', file.read(4))[0]
    
    return file.read(length).decode('utf-8', errors='replace')


def read_reals(file: BinaryIO, dtype: np.dtype, count: int = 1) -> np.ndarray:
    """
    Read an array of reals from a binary file
    
    Parameters:
    -----------
    file : BinaryIO
        Open file handle
    dtype : np.dtype
        Data type to read
    count : int, default=1
        Number of values to read
        
    Returns:
    --------
    np.ndarray
        Array of read values
    """
    width = np.dtype(dtype).itemsize
    
    # Check if we're aligned properly for memory mapping
    if file.tell() % width == 0:
        # Create a memory-mapped array
        try:
            mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            result = np.ndarray(
                shape=(count,), 
                dtype=dtype, 
                buffer=mm, 
                offset=file.tell()
            )
            file.seek(file.tell() + width * count)
            return np.copy(result)  # Copy to avoid issues with the mmap later
        except (ValueError, TypeError, OSError):
            # Fall back to fromfile if memory mapping fails
            pass
    
    # Read directly
    data = file.read(width * count)
    return np.frombuffer(data, dtype=dtype)


def to_datetime(floats: np.ndarray) -> np.ndarray:
    """
    Convert float64 values to datetime objects.
    JMP uses the 1904 date system (seconds since Jan 1, 1904).
    
    Parameters:
    -----------
    floats : np.ndarray
        Array of float values representing seconds since JMP's epoch
        
    Returns:
    --------
    np.ndarray
        Array of datetime64 values
    """
    # Create a datetime64 array with NaT for NaN values
    result = np.empty(floats.shape, dtype='datetime64[s]')
    result[:] = np.datetime64('NaT')
    
    # Get mask of non-NaN values
    mask = ~np.isnan(floats)
    if not np.any(mask):
        return result
    
    # Convert seconds since JMP epoch to Unix timestamp
    unix_epoch = datetime(1970, 1, 1)
    seconds_offset = int((JMP_STARTDATE - unix_epoch).total_seconds())
    
    # Convert valid values
    unix_times = floats[mask] + seconds_offset
    result[mask] = np.array([np.datetime64(int(t), 's') for t in unix_times])
    
    return result


def check_magic(file: BinaryIO) -> bool:
    """
    Check if the file has the correct JMP magic bytes
    
    Parameters:
    -----------
    file : BinaryIO
        Open file handle
        
    Returns:
    --------
    bool
        True if the file has the correct magic bytes, False otherwise
    """
    current_pos = file.tell()
    file.seek(0)
    magic = file.read(len(MAGIC_JMP))
    file.seek(current_pos)  # Restore position
    return magic == MAGIC_JMP


def sentinel_to_missing(data: np.ndarray) -> pd.Series:
    """
    Convert sentinel values in numeric arrays to pandas missing values
    
    Parameters:
    -----------
    data : np.ndarray
        Input array potentially containing sentinel values
        
    Returns:
    --------
    pd.Series
        Series with sentinel values converted to pandas missing values
    """
    dtype = data.dtype
    
    # Handle float data
    if np.issubdtype(dtype, np.floating):
        return pd.Series(data)  # pandas handles NaN automatically
    
    # Handle integer types
    if np.issubdtype(dtype, np.integer):
        # Define sentinel values for different integer types
        if dtype == np.int8:
            sentinel = np.iinfo(np.int8).min + 1
            pd_type = 'Int8'
        elif dtype == np.int16:
            sentinel = np.iinfo(np.int16).min + 1
            pd_type = 'Int16'
        elif dtype == np.int32:
            sentinel = np.iinfo(np.int32).min + 1 
            pd_type = 'Int32'
        else:
            return pd.Series(data)  # No sentinel for other types
        
        # Check if data contains the sentinel value
        has_sentinel = np.any(data == sentinel)
        
        if has_sentinel:
            # Create a pandas Series with missing value support
            series = pd.Series(data)
            # Convert sentinel to NaN
            series = series.replace(sentinel, np.nan)
            # Convert to pandas nullable integer type
            return series.astype(pd_type)
    
    # Default case: return as regular Series
    return pd.Series(data)


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert hex color code to RGB tuple with values in [0, 1]
    
    Parameters:
    -----------
    hex_color : str
        Hex color code, e.g., '#FF00FF'
        
    Returns:
    --------
    Tuple[float, float, float]
        RGB tuple with values in range [0, 1]
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)


def bit_cat(a: int, b: int) -> int:
    """
    Concatenate two 8-bit values into a single 16-bit value
    
    Parameters:
    -----------
    a : int
        High-order byte
    b : int
        Low-order byte
        
    Returns:
    --------
    int
        16-bit integer combining the two bytes
    """
    return (a << 8) | b