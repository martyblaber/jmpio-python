"""
Utility functions for reading and writing JMP files
"""

import struct
from datetime import datetime
from typing import BinaryIO, TypeVar

import numpy as np
import pandas as pd

from .constants import JMP_STARTDATE, MAGIC_JMP

T = TypeVar("T")


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
        length = struct.unpack("b", file.read(1))[0]
    elif width == 2:
        length = struct.unpack("h", file.read(2))[0]
    else:  # width == 4
        length = struct.unpack("i", file.read(4))[0]

    return file.read(length).decode("utf-8", errors="replace")


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

    # Read directly using file.read() and then convert using np.frombuffer()
    data = file.read(width * count)

    # Check if the expected number of bytes was read
    if len(data) < width * count:
        raise EOFError(
            f"Attempted to read {width * count} bytes for {count} element(s) of type {dtype}, "
            f"but only received {len(data)} bytes. End of file reached prematurely."
        )

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
    result = np.empty(floats.shape, dtype="datetime64[s]")
    result[:] = np.datetime64("NaT")

    # Get mask of non-NaN values
    mask = ~np.isnan(floats)
    if not np.any(mask):
        return result

    # Convert seconds since JMP epoch to Unix timestamp
    unix_epoch = datetime(1970, 1, 1)
    seconds_offset = int((JMP_STARTDATE - unix_epoch).total_seconds())

    # Convert valid values
    unix_times = floats[mask] + seconds_offset
    result[mask] = np.array([np.datetime64(int(t), "s") for t in unix_times])

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
            pd_type = "Int8"
        elif dtype == np.int16:
            sentinel = np.iinfo(np.int16).min + 1
            pd_type = "Int16"
        elif dtype == np.int32:
            sentinel = np.iinfo(np.int32).min + 1
            pd_type = "Int32"
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


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """
    Convert hex color code to RGB tuple with values in [0, 1]

    Parameters:
    -----------
    hex_color : str
        Hex color code, e.g., '#FF00FF'

    Returns:
    --------
    tuple[float, float, float]
        RGB tuple with values in range [0, 1]
    """
    hex_color = hex_color.lstrip("#")
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


def find_sequence_in_file(
    file: BinaryIO,
    seq: bytes,
    start: int = 0,
    stop: int = -1,
    chunk_size: int = 4096,
) -> int | None:
    """
    Search for a byte sequence in a BinaryIO stream using chunked reads with overlap.

    This function reads the file in 4KB chunks (by default) and ensures that
    matches that span chunk boundaries are detected by overlapping the last
    len(seq)-1 bytes of the previous chunk with the next chunk.

    Parameters:
    -----------
    file : BinaryIO
        Open file-like object positioned anywhere (position will be restored).
    seq : bytes
        Byte sequence to search for. Must be non-empty.
    start : int, default=0
        Start byte offset (inclusive).
    stop : int, default=-1
        Stop byte offset (exclusive). Use -1 to indicate end-of-file.
    chunk_size : int, default=4096
        Size of chunks to read.

    Returns:
    --------
    int | None
        The absolute byte offset of the first match, or None if not found.
    """
    if not isinstance(seq, (bytes, bytearray)) or len(seq) == 0:
        return None

    # Preserve caller's position
    orig_pos = file.tell()
    try:
        # Determine bounds
        file.seek(0, 2)
        file_size = file.tell()
        if stop < 0 or stop > file_size:
            stop = file_size

        if start < 0:
            start = 0
        if start >= stop:
            return None

        # Begin scanning
        file.seek(start)
        pos = start
        overlap_len = max(0, len(seq) - 1)
        tail = b""

        while pos < stop:
            to_read = min(chunk_size, stop - pos)
            if to_read <= 0:
                break
            chunk = file.read(to_read)
            if not chunk:
                break

            search_buf = tail + chunk
            local_idx = search_buf.find(seq)
            if local_idx != -1:
                # Compute absolute index considering the tail length
                abs_idx = pos - len(tail) + local_idx
                return abs_idx

            # Prepare tail for next iteration
            if overlap_len > 0:
                tail = search_buf[-overlap_len:]
            else:
                tail = b""

            pos += len(chunk)
        return None
    finally:
        # Restore caller's position
        try:
            file.seek(orig_pos)
        except Exception:
            pass
