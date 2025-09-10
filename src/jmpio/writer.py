"""
Functions for writing JMP files
"""

import gzip
import os
import struct
from datetime import date, datetime, time, timedelta
from typing import BinaryIO

import numpy as np
import pandas as pd

from .constants import (
    GZIP_SECTION_START,
    JMP_STARTDATE,
    MAGIC_JMP,
    ROWSTATE_COLORS,
    ROWSTATE_MARKERS,
)
from .types import RowState


def write_jmp(df: pd.DataFrame, filename: str, compress: bool = True, version: str = "16.0") -> None:
    """
    Write a pandas DataFrame to a JMP file

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to write
    filename : str
        Path to the output file
    compress : bool, default=True
        Whether to compress the data
    version : str, default="16.0"
        JMP version to use in the file header

    Returns:
    --------
    None

    Examples:
    ---------
    >>> import pandas as pd
    >>> import jmpio
    >>>
    >>> # Create a DataFrame
    >>> df = pd.DataFrame({
    >>>     'ints': [1, 2, 3, 4],
    >>>     'floats': [1.1, 2.2, 3.3, 4.4],
    >>>     'strings': ['a', 'bb', 'ccc', 'dddd']
    >>> })
    >>>
    >>> # Write to a JMP file
    >>> jmpio.write_jmp(df, 'output.jmp')
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(os.path.abspath(filename))
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Open file in binary write mode
    with open(filename, "wb") as file:
        # Write file header
        write_file_header(file, df, version)

        # Write column metadata
        column_offsets = write_column_metadata(file, df)

        # Write column data
        for i, column_name in enumerate(df.columns):
            column_data = df[column_name]
            write_column_data(file, column_data, column_offsets[i], column_name, compress)

        # Any final corrections or clean-up
        finalize_file(file)


def write_file_header(file: BinaryIO, df: pd.DataFrame, version: str) -> None:
    """
    Write JMP file header

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    df : pd.DataFrame
        DataFrame being written
    version : str
        JMP version to use in the header
    """
    # Write magic bytes (signature)
    file.write(MAGIC_JMP)

    # Write padding up to the row offset
    padding_size = 368 - len(MAGIC_JMP)
    padding_data = bytearray([0] * padding_size)

    # Add some metadata in the padding (this is reverse-engineered)
    # Here we could add metadata like creation software, etc.
    file.write(padding_data)

    # Write number of rows (Int64) and columns (Int32)
    file.write(struct.pack("<q", len(df)))
    file.write(struct.pack("<i", len(df.columns)))

    # Write unknown values (5 Int16s)
    for _ in range(5):
        file.write(struct.pack("<h", 0))

    # Write character set (typically ASCII or UTF-8)
    charset = "UTF8"
    file.write(struct.pack("<b", len(charset)))
    file.write(charset.encode("utf-8"))
    file.write(b"\0")  # Null terminator

    # Write some unknown values (3 UInt16s)
    for _ in range(3):
        file.write(struct.pack("<H", 0))

    # Write save time (current time as seconds since JMP epoch)
    current_time = datetime.now()
    seconds_since_epoch = (current_time - JMP_STARTDATE).total_seconds()
    file.write(struct.pack("<d", seconds_since_epoch))

    # Write more unknown values (1 UInt16)
    file.write(struct.pack("<H", 18))  # From observed files

    # Write build string with version
    build_string = f"JMP Version {version}"
    file.write(struct.pack("<i", len(build_string)))
    file.write(build_string.encode("utf-8"))


def write_column_metadata(file: BinaryIO, df: pd.DataFrame) -> list[int]:
    """
    Write metadata about columns

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    df : pd.DataFrame
        DataFrame being written

    Returns:
    --------
    list[int]
        List of file offsets for each column's data
    """
    # Write column metadata section marker
    file.write(b"\xff\xff")

    # Write some zeros (observed format)
    file.write(struct.pack("<QQQQ", 0, 0, 0, 0))

    # Write visible and hidden column counts
    # For now, all columns are visible
    n_visible = len(df.columns)
    n_hidden = 0
    file.write(struct.pack("<II", n_visible, n_hidden))

    # Write some unknown values
    file.write(struct.pack("<II", 0, 0))

    # Write visible column indices (0-based)
    for i in range(n_visible):
        file.write(struct.pack("<I", i))

    # Write column widths (display width in JMP)
    for column_name in df.columns:
        # Default column width is based on name length
        width = min(max(len(column_name) * 8, 40), 300)
        file.write(struct.pack("<H", width))

    # Write some more unknown values
    for _ in range(7):
        file.write(struct.pack("<I", 0))

    # Mark beginning of column data section
    column_section_pos = file.tell()

    # Write the number of columns again as a check
    file.write(struct.pack("<i", len(df.columns)))

    # Reserve space for column offsets
    offset_pos = file.tell()
    for _ in range(len(df.columns)):
        file.write(struct.pack("<q", 0))  # Placeholder for column offsets

    # Calculate the actual column offsets
    # Start position for first column's data
    data_start_pos = file.tell() + 1000  # Arbitrary buffer

    # Align data start to 8-byte boundary for better performance
    data_start_pos = (data_start_pos + 7) & ~7

    # Seek to position to store column data
    file.seek(data_start_pos)

    # Calculate offsets for each column
    column_offsets = []
    for i, column_name in enumerate(df.columns):
        column_offsets.append(file.tell())

        # Skip ahead a reasonable amount to make room for column headers
        # We'll come back and fill this in properly
        file.write(bytearray([0] * 100))  # Arbitrary space for column header

    # Go back and write the actual offsets
    file.seek(offset_pos)
    for offset in column_offsets:
        file.write(struct.pack("<q", offset))

    # Return to last position
    file.seek(column_offsets[-1] + 100)

    return column_offsets


def write_column_data(
    file: BinaryIO,
    column: pd.Series,
    offset: int,
    column_name: str,
    compress: bool = True,
) -> None:
    """
    Write a single column's data to the file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    offset : int
        File offset where this column should start
    column_name : str
        Name of the column
    compress : bool, default=True
        Whether to compress the data
    """
    # Save the current position so we can return to it
    current_pos = file.tell()

    # Seek to the column's offset
    file.seek(offset)

    # Write column name
    file.write(struct.pack("<h", len(column_name)))
    file.write(column_name.encode("utf-8"))

    # Determine data type and write appropriate type markers
    data_type = get_column_data_type(column)

    if data_type == "float":
        write_float_column(file, column, compress)
    elif data_type == "int":
        write_int_column(file, column, compress)
    elif data_type == "string":
        write_string_column(file, column, compress)
    elif data_type == "datetime":
        write_datetime_column(file, column, compress)
    elif data_type == "date":
        write_date_column(file, column, compress)
    elif data_type == "time":
        write_time_column(file, column, compress)
    elif data_type == "duration":
        write_duration_column(file, column, compress)
    elif data_type == "rowstate":
        write_rowstate_column(file, column, compress)
    else:
        # Default to writing as string
        write_string_column(file, column, compress)

    # Return to the previous position
    file.seek(current_pos)


def get_column_data_type(column: pd.Series) -> str:
    """
    Determine the JMP data type for a pandas column

    Parameters:
    -----------
    column : pd.Series
        Column to analyze

    Returns:
    --------
    str
        String identifier of the column type
    """
    # Get pandas dtype
    dtype = column.dtype

    # Check for pandas extension types
    if hasattr(dtype, "name"):
        dtype_name = dtype.name

        # Check for pandas nullable integer types
        if dtype_name in ["Int8", "Int16", "Int32", "Int64"]:
            return "int"

        # Check for pandas datetime types
        if dtype_name.startswith("datetime"):
            # Try to determine if it's a date or datetime
            try:
                if all(pd.notna(t) and t.time() == time(0, 0) for t in pd.to_datetime(column.dropna())):
                    return "date"
                return "datetime"
            except (AttributeError, TypeError):
                return "datetime"

    # Check numpy/pandas basic types
    if np.issubdtype(dtype, np.integer):
        return "int"
    elif np.issubdtype(dtype, np.floating):
        return "float"
    elif np.issubdtype(dtype, np.datetime64):
        # Try to distinguish date from datetime
        try:
            # Check if all values have time component equal to midnight
            has_time = False
            for val in pd.to_datetime(column.dropna()):
                time_part = val.time()
                if time_part.hour != 0 or time_part.minute != 0 or time_part.second != 0:
                    has_time = True
                    break

            if has_time:
                return "datetime"
            else:
                return "date"
        except (AttributeError, TypeError):
            return "datetime"
    elif np.issubdtype(dtype, np.timedelta64):
        return "duration"
    elif dtype == "object":
        # Check first non-null value type
        non_null = column.dropna()
        if len(non_null) == 0:
            return "string"  # Default for empty series

        sample = non_null.iloc[0]

        if isinstance(sample, str):
            return "string"
        elif isinstance(sample, (datetime, np.datetime64)):
            return "datetime"
        elif isinstance(sample, date) and not isinstance(sample, datetime):
            return "date"
        elif isinstance(sample, time):
            return "time"
        elif isinstance(sample, timedelta):
            return "duration"
        elif isinstance(sample, RowState):
            return "rowstate"

    # Default to string for any other types
    return "string"


def write_float_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write a float column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    compress : bool
        Whether to compress the data
    """
    # Type markers for float64
    dt1 = 0x09 if compress else 0x01  # 0x09 for compressed data
    dt2, dt3, dt4, dt5, dt6 = 0x01, 0x00, 0x00, 0x00, 0x08
    file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

    # Convert to numpy array, handling missing values
    data = column.fillna(np.nan).to_numpy(dtype=np.float64)

    if compress:
        # Write some header data for compressed
        file.write(bytearray([0] * 8))

        # Compress the data
        compressed_data = gzip.compress(data.tobytes())

        # Write gzip section marker
        file.write(GZIP_SECTION_START)

        # Write compressed and uncompressed sizes
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(data) * 8))  # uncompressed size (8 bytes per float64)

        # Write the compressed data
        file.write(compressed_data)
    else:
        # Write the raw data directly
        file.write(data.tobytes())


def write_int_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write an integer column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    compress : bool
        Whether to compress the data
    """
    # Determine integer size
    dtype_name = str(column.dtype).lower()

    if "int8" in dtype_name:
        element_size = 1
        np_dtype = np.int8
        dt6 = 0x01
    elif "int16" in dtype_name:
        element_size = 2
        np_dtype = np.int16
        dt6 = 0x02
    elif "int32" in dtype_name:
        element_size = 4
        np_dtype = np.int32
        dt6 = 0x04
    else:
        # Default to int8 for other integer types if values are small enough
        min_val = column.min() if not column.empty and not column.isna().all() else 0
        max_val = column.max() if not column.empty and not column.isna().all() else 0

        if min_val >= -128 and max_val <= 127:
            element_size = 1
            np_dtype = np.int8
            dt6 = 0x01
        elif min_val >= -32768 and max_val <= 32767:
            element_size = 2
            np_dtype = np.int16
            dt6 = 0x02
        else:
            element_size = 4
            np_dtype = np.int32
            dt6 = 0x04

    # Type markers for integer
    dt1 = 0x09 if compress else 0x01  # 0x09 for compressed data
    dt2, dt3, dt4, dt5 = 0x01, 0x00, 0x00, 0x00
    file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

    # For missing values, use dtype's minimum value + 1 as sentinel
    sentinel = np.iinfo(np_dtype).min + 1

    # Convert to numpy array, replacing NaN with sentinel
    data = column.fillna(sentinel).to_numpy(dtype=np_dtype)

    if compress:
        # Write some header data for compressed columns
        file.write(bytearray([0] * 8))

        # Compress the data
        compressed_data = gzip.compress(data.tobytes())

        # Write gzip section marker
        file.write(GZIP_SECTION_START)

        # Write compressed and uncompressed sizes
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(data) * element_size))  # uncompressed size

        # Write the compressed data
        file.write(compressed_data)
    else:
        # Write the raw data directly
        file.write(data.tobytes())


def write_string_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write a string column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    compress : bool
        Whether to compress the data
    """
    # Replace NaN with empty string
    column = column.fillna("")

    # Get string lengths
    string_lengths = column.str.len()
    max_length = string_lengths.max()
    min_length = string_lengths.min()

    # Determine if we should use fixed width or variable width
    use_fixed_width = max_length == min_length and max_length > 0

    if use_fixed_width:
        # For fixed width strings
        dt1 = 0x09 if compress else 0x02  # 0x09 for compressed data
        dt2, dt3, dt4 = 0x01, 0x00, 0x00
        dt5 = min(max_length, 65535)  # Max width is UInt16 max
        dt6 = 0x00

        file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

        # Create a byte array of fixed width strings
        string_data = bytearray()
        for s in column:
            # Encode the string as UTF-8 and pad/truncate to fixed width
            encoded = s.encode("utf-8")
            padded = encoded[:dt5].ljust(dt5, b"\0")
            string_data.extend(padded)

        if compress:
            # Write some header for compressed data
            file.write(bytearray([0] * 8))

            # Compress the string data
            compressed_data = gzip.compress(string_data)

            # Write gzip section marker
            file.write(GZIP_SECTION_START)

            # Write compressed and uncompressed sizes
            file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
            file.write(struct.pack("<Q", len(string_data)))  # uncompressed size

            # Write the compressed data
            file.write(compressed_data)
        else:
            # Write raw string data
            file.write(string_data)
    else:
        # For variable width strings
        dt1 = 0x09  # Always use compression for variable width
        dt2, dt3, dt4, dt5, dt6 = 0x01, 0x00, 0x00, 0x00, 0x00

        file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

        # Prepare variable width string data

        # Determine width needed for length values
        use_int16 = max_length >= 128

        # Prepare header
        header = bytearray([0] * 13)
        header[9] = 2 if use_int16 else 1  # Width bytes

        # Prepare lengths and string data
        lengths = bytearray()
        string_data = bytearray()

        for s in column:
            encoded = s.encode("utf-8")
            string_data.extend(encoded)

            if use_int16:
                lengths.extend(struct.pack("<h", len(encoded)))
            else:
                lengths.extend(struct.pack("<b", len(encoded)))

        # Combine all data
        all_data = header + lengths + string_data

        # Always compress variable width strings
        compressed_data = gzip.compress(all_data)

        # Write gzip section marker
        file.write(GZIP_SECTION_START)

        # Write compressed and uncompressed sizes
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(all_data)))  # uncompressed size

        # Write the compressed data
        file.write(compressed_data)


def write_datetime_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write a datetime column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    compress : bool
        Whether to compress the data
    """
    # Type markers for datetime
    dt1 = 0x09 if compress else 0x01  # 0x09 for compressed data
    dt2, dt3, dt4, dt5, dt6 = 0x01, 0x00, 0x69, 0x69, 0x08
    file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

    # Convert to seconds since JMP epoch
    def to_jmp_time(dt):
        if pd.isna(dt):
            return np.nan

        if isinstance(dt, (np.datetime64, pd.Timestamp)):
            dt = pd.Timestamp(dt).to_pydatetime()

        return (dt - JMP_STARTDATE).total_seconds()

    epoch_seconds = column.apply(to_jmp_time)

    # Convert to numpy array
    data = epoch_seconds.to_numpy(dtype=np.float64)

    if compress:
        # Write some header for compressed data
        file.write(bytearray([0] * 8))

        # Compress the data
        compressed_data = gzip.compress(data.tobytes())

        # Write gzip section marker
        file.write(GZIP_SECTION_START)

        # Write compressed and uncompressed sizes
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(data) * 8))  # uncompressed size (8 bytes per float64)

        # Write the compressed data
        file.write(compressed_data)
    else:
        # Write the raw data
        file.write(data.tobytes())


def write_date_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write a date column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    compress : bool
        Whether to compress the data
    """
    # Type markers for date
    dt1 = 0x09 if compress else 0x01  # 0x09 for compressed data
    dt2, dt3, dt4, dt5, dt6 = 0x01, 0x00, 0x65, 0x65, 0x08
    file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

    # Convert to seconds since JMP epoch (with time part set to 00:00:00)
    def to_jmp_date(d):
        if pd.isna(d):
            return np.nan

        if isinstance(d, (np.datetime64, pd.Timestamp)):
            d = pd.Timestamp(d).to_pydatetime().date()
        elif isinstance(d, datetime):
            d = d.date()

        return (datetime.combine(d, time.min) - JMP_STARTDATE).total_seconds()

    epoch_seconds = column.apply(to_jmp_date)

    # Convert to numpy array
    data = epoch_seconds.to_numpy(dtype=np.float64)

    if compress:
        # Write some header for compressed data
        file.write(bytearray([0] * 8))

        # Compress the data
        compressed_data = gzip.compress(data.tobytes())

        # Write gzip section marker
        file.write(GZIP_SECTION_START)

        # Write compressed and uncompressed sizes
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(data) * 8))  # uncompressed size (8 bytes per float64)

        # Write the compressed data
        file.write(compressed_data)
    else:
        # Write the raw data
        file.write(data.tobytes())


def write_time_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write a time column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    compress : bool
        Whether to compress the data
    """
    # Type markers for time
    dt1 = 0x09 if compress else 0x01  # 0x09 for compressed data
    dt2, dt3, dt4, dt5, dt6 = 0x01, 0x00, 0x82, 0x82, 0x08
    file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

    # Convert time to seconds since midnight
    def to_seconds(t):
        if pd.isna(t):
            return np.nan

        if isinstance(t, (datetime, pd.Timestamp)):
            t = t.time()

        return t.hour * 3600 + t.minute * 60 + t.second

    seconds = column.apply(to_seconds)

    # Convert to numpy array
    data = seconds.to_numpy(dtype=np.float64)

    if compress:
        # Write some header for compressed data
        file.write(bytearray([0] * 8))

        # Compress the data
        compressed_data = gzip.compress(data.tobytes())

        # Write gzip section marker
        file.write(GZIP_SECTION_START)

        # Write compressed and uncompressed sizes
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(data) * 8))  # uncompressed size (8 bytes per float64)

        # Write the compressed data
        file.write(compressed_data)
    else:
        # Write the raw data
        file.write(data.tobytes())


def write_duration_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write a duration column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write
    compress : bool
        Whether to compress the data
    """
    # Type markers for duration
    dt1 = 0x09 if compress else 0x01  # 0x09 for compressed data
    dt2, dt3, dt4, dt5, dt6 = 0x01, 0x00, 0x6C, 0x6C, 0x08
    file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

    # Convert duration to seconds
    def to_seconds(d):
        if pd.isna(d):
            return np.nan

        if isinstance(d, (timedelta, pd.Timedelta)):
            return d.total_seconds()

        return float(d)  # Attempt to convert other types

    seconds = column.apply(to_seconds)

    # Convert to numpy array
    data = seconds.to_numpy(dtype=np.float64)

    if compress:
        # Write some header for compressed data
        file.write(bytearray([0] * 8))

        # Compress the data
        compressed_data = gzip.compress(data.tobytes())

        # Write gzip section marker
        file.write(GZIP_SECTION_START)

        # Write compressed and uncompressed sizes
        file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
        file.write(struct.pack("<Q", len(data) * 8))  # uncompressed size (8 bytes per float64)

        # Write the compressed data
        file.write(compressed_data)
    else:
        # Write the raw data
        file.write(data.tobytes())


def write_rowstate_column(file: BinaryIO, column: pd.Series, compress: bool) -> None:
    """
    Write a row state column to a JMP file

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    column : pd.Series
        Column data to write (containing RowState objects)
    compress : bool
        Whether to compress the data
    """
    # Type markers for row state
    dt1 = 0x09  # Always use compressed format for row states
    dt2, dt3, dt4 = 0x03, 0x00, 0x00
    dt5 = 8  # Width of each row state entry
    dt6 = 0x00
    file.write(struct.pack("<BBBBBB", dt1, dt2, dt3, dt4, dt5, dt6))

    # Prepare row state data
    row_data = bytearray()

    for rs in column:
        if pd.isna(rs):
            # Default row state for missing values
            marker_idx = 0  # First marker
            color_idx = 0  # First color (black)
            data_entry = bytearray([0, color_idx, 0, 0, 0, 0, 0, marker_idx])
            row_data.extend(data_entry)
            continue

        # Get marker index
        try:
            marker_idx = ROWSTATE_MARKERS.index(rs.marker)

            # Write as UInt16 in little-endian format (lower byte first)
            marker_bytes = marker_idx.to_bytes(2, byteorder="little")
        except ValueError:
            # If marker not found in predefined list, use character code
            marker_bytes = ord(rs.marker).to_bytes(2, byteorder="little")

        # Get RGB color components
        r, g, b = rs.color

        # Check if the color is in the predefined list
        hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}".upper()
        try:
            color_idx = ROWSTATE_COLORS.index(hex_color)
            custom_color = False
        except ValueError:
            # Use custom color encoding
            color_idx = 0
            custom_color = True

        # Build row state entry
        if custom_color:
            # Custom color format
            data_entry = bytearray(
                [
                    0,  # Unknown
                    int(b * 255),  # Blue
                    int(g * 255),  # Green
                    int(r * 255),  # Red
                    0xFF,  # Flag for custom color
                    0,  # Unknown
                    marker_bytes[0],  # Marker (low byte)
                    marker_bytes[1],  # Marker (high byte)
                ]
            )
        else:
            # Predefined color format
            data_entry = bytearray(
                [
                    0,  # Unknown
                    color_idx,  # Color index
                    0,  # Unknown
                    0,  # Unknown
                    0,  # Flag for predefined color
                    0,  # Unknown
                    marker_bytes[0],  # Marker (low byte)
                    marker_bytes[1],  # Marker (high byte)
                ]
            )

        row_data.extend(data_entry)

    # Compress row state data
    compressed_data = gzip.compress(row_data)

    # Write some header for compressed data
    file.write(bytearray([0] * 8))

    # Write gzip section marker
    file.write(GZIP_SECTION_START)

    # Write compressed and uncompressed sizes
    file.write(struct.pack("<Q", len(compressed_data)))  # compressed size
    file.write(struct.pack("<Q", len(row_data)))  # uncompressed size

    # Write the compressed data
    file.write(compressed_data)


def finalize_file(file: BinaryIO) -> None:
    """
    Perform any final operations on the file before closing

    Parameters:
    -----------
    file : BinaryIO
        Open file handle for writing
    """
    # This function can be used for any final adjustments to the file
    # Ensure file is properly aligned on 8-byte boundary
    current_pos = file.tell()
    padding = 8 - (current_pos % 8) if current_pos % 8 != 0 else 0

    if padding > 0:
        file.write(bytearray([0] * padding))

    # We may need to update some file sizes or checksums here
    # For now, no additional finalization is needed
