"""
Functions for reading column data from JMP files
"""

import struct
import gzip
import io
from typing import BinaryIO, List, Dict, Any, Union, Optional, Tuple, cast
import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta

from .constants import GZIP_SECTION_START, JMP_STARTDATE, ROWSTATE_MARKERS, ROWSTATE_COLORS
from .types import JMPInfo, RowState
from .utils import read_string, read_reals, to_datetime, sentinel_to_missing, bit_cat


def read_column_data(file: BinaryIO, info: JMPInfo, column_idx: int) -> Any:
    """
    Read data from a specific column in a JMP file
    
    Parameters:
    -----------
    file : BinaryIO
        Open file handle to a JMP file
    info : JMPInfo
        JMP file metadata
    column_idx : int
        Index of the column to read
        
    Returns:
    --------
    Any
        Column data as a numpy array or pandas Series
    """
    if not (0 <= column_idx < info.ncols):
        raise ValueError(f"Column index {column_idx} is out of bounds (0-{info.ncols-1})")
    
    # Seek to the column offset
    file.seek(info.column.offsets[column_idx])
    
    # Read column name
    column_name = read_string(file, 2)
    column_name_len = len(column_name.encode('utf-8'))  # Get actual bytes used
    
    # Read column data type markers
    dt = struct.unpack('BBBBBB', file.read(6))
    dt1, dt2, dt3, dt4, dt5, dt6 = dt
    
    start_pos = file.tell()
    
    # Handle compressed data
    is_compressed = dt1 in [0x09, 0x0a]
    
    if is_compressed:
        # Find the start of the gzip section
        while True:
            bytes_read = file.read(4)
            if not bytes_read:
                raise EOFError(f"End of file reached before finding GZIP_SECTION_START in column {column_name}")
            if bytes_read == GZIP_SECTION_START:
                break
        
        # Read compressed and uncompressed sizes
        gzip_len = struct.unpack('Q', file.read(8))[0]  # UInt64
        gunzip_len = struct.unpack('Q', file.read(8))[0]  # UInt64
        
        # Read compressed data
        compressed_data = file.read(gzip_len)
        
        # Decompress
        try:
            decompressed_data = gzip.decompress(compressed_data)
            buffer = io.BytesIO(decompressed_data)
        except Exception as e:
            raise ValueError(f"Failed to decompress data for column '{column_name}': {str(e)}")
    else:
        # For uncompressed data, figure out column end
        if column_idx == info.ncols - 1:
            # Last column, read to end of file
            file.seek(0, 2)  # Seek to end
            col_end = file.tell()
        else:
            col_end = info.column.offsets[column_idx + 1]
        
        # Read all column data
        column_size = col_end - info.column.offsets[column_idx]
        file.seek(info.column.offsets[column_idx])
        column_data = file.read(column_size)
        
        # Create in-memory buffer
        buffer = io.BytesIO(column_data)
        
        # Skip the header we've already read (column name + length bytes + 6 dtype bytes)
        buffer.seek(2 + column_name_len + 6)
    
    try:
        # Numeric data types (float, integers, dates, times)
        if dt1 in [0x01, 0x0a]:
            # Determine data type
            if dt6 == 0x01:
                dtype = np.int8
            elif dt6 == 0x02:
                dtype = np.int16
            elif dt6 == 0x04:
                dtype = np.int32
            else:
                dtype = np.float64
            
            # For compressed data, data is at the end of the decompressed buffer
            if is_compressed:
                buffer.seek(-dtype().itemsize * info.nrows, 2)  # Seek from end
                
            raw_data = buffer.read(dtype().itemsize * info.nrows)
            data_array = np.frombuffer(raw_data, dtype=dtype)
            
            # Convert sentinel values to missing
            data_array = sentinel_to_missing(data_array)
            
            # Check specific type markers for different numeric types
            
            # Regular numeric values (Float64 or byte integers)
            if ((dt4 == dt5 and dt4 in [
                    0x00, 0x03, 0x42, 0x43, 0x44, 0x59, 0x60, 0x63
                ]) or
                dt5 in [0x5e, 0x63]):  # fixed decimal, dt3=width, dt4=decimal places
                return pd.Series(data_array)
            
            # Currency
            if dt4 == dt5 and dt4 in [0x5f]:
                return pd.Series(data_array)
            
            # Longitude
            if dt4 == dt5 and dt4 in [0x54, 0x55, 0x56]:
                return pd.Series(data_array)
            
            # Latitude
            if dt4 == dt5 and dt4 in [0x51, 0x52, 0x53]:
                return pd.Series(data_array)
            
            # For date/time values, convert to appropriate datetime format
            # First convert to datetime64
            datetime_array = to_datetime(data_array)
            
            # Date
            if ((dt4 == dt5 and dt4 in [
                    0x65, 0x66, 0x67, 0x6e, 0x6f, 0x70, 0x71, 0x72, 0x75, 0x76, 0x7a,
                    0x7f, 0x88, 0x8b,
                ]) or
                [dt4, dt5] in [[0x67, 0x65], [0x6f, 0x65], [0x72, 0x65], [0x72, 0x6f], 
                            [0x72, 0x7f], [0x72, 0x80], [0x7f, 0x72], [0x88, 0x65], [0x88, 0x7a]]):
                return pd.Series(datetime_array.astype('datetime64[D]'))
            
            # DateTime
            if (dt5 in [0x69, 0x6a, 0x73, 0x74, 0x77, 0x78, 0x7e, 0x81] and dt4 in [
                    0x69, 0x6a, 0x6c, 0x6d, 0x73, 0x74, 0x77, 0x78, 0x79, 0x7b, 0x7c,
                    0x7d, 0x7e, 0x80, 0x81, 0x82, 0x86, 0x87, 0x89, 0x8a,
                ] or
                dt4 == dt5 in [0x79, 0x7d] or
                [dt4, dt5] in [[0x77, 0x80], [0x77, 0x7f], [0x89, 0x65]]):
                return pd.Series(datetime_array)
            
            # Time
            if dt4 == dt5 in [0x82]:
                # Extract just the time portion of the datetime
                return pd.Series(pd.to_datetime(datetime_array).dt.time)
            
            # Duration
            if ((dt4 == dt5 and dt4 in [
                    0x0c, 0x6b, 0x6c, 0x6d, 0x83, 0x84, 0x85
                ]) or
                [dt4, dt5] in [[0x84, 0x79]]):
                # Calculate duration in milliseconds from JMP_STARTDATE
                epoch_start = np.datetime64(JMP_STARTDATE)
                delta_ms = (datetime_array - epoch_start) / np.timedelta64(1, 'ms')
                return pd.Series(pd.to_timedelta(delta_ms, unit='ms'))
        
        # Alternative byte integer encoding
        if dt1 in [0xff, 0xfe, 0xfc]:
            if dt5 == 0x01:
                dtype = np.int8
            elif dt5 == 0x02:
                dtype = np.int16
            elif dt5 == 0x04:
                dtype = np.int32
            else:
                dtype = np.float64
            
            # For compressed data or at the end of the file
            if is_compressed:
                buffer.seek(-dtype().itemsize * info.nrows, 2)  # From end
            
            raw_data = buffer.read(dtype().itemsize * info.nrows)
            data_array = np.frombuffer(raw_data, dtype=dtype)
            
            # Convert sentinel values to missing
            data_array = sentinel_to_missing(data_array)
            return pd.Series(data_array)
        
        # Row states (markers and colors)
        if dt1 == 0x09 and dt2 == 0x03:
            width = dt5
            row_states = []
            
            # For compressed data, the row states should be at the end
            if is_compressed:
                buffer.seek(-width * info.nrows, 2)  # From end
            
            # Read each row state
            for row in range(info.nrows):
                row_data = buffer.read(width)
                if len(row_data) < width:
                    raise ValueError(f"Not enough data for row states in column {column_name}")
                
                marker_idx = bit_cat(row_data[7], row_data[6])
                marker = ROWSTATE_MARKERS[marker_idx] if marker_idx < len(ROWSTATE_MARKERS) else chr(marker_idx)
                
                # Extract color
                if row_data[4] == 0xff:
                    r, g, b = row_data[3] / 255, row_data[2] / 255, row_data[1] / 255
                else:
                    # Get color from predefined colors
                    color_idx = row_data[1]
                    if 0 <= color_idx < len(ROWSTATE_COLORS):
                        color_hex = ROWSTATE_COLORS[color_idx]
                        r = int(color_hex[1:3], 16) / 255
                        g = int(color_hex[3:5], 16) / 255
                        b = int(color_hex[5:7], 16) / 255
                    else:
                        # Default to black if color index is out of range
                        r, g, b = 0, 0, 0
                
                row_states.append(RowState(marker=marker, color=(r, g, b)))
            
            return pd.Series(row_states)
        
        # Character data
        if dt1 in [0x02, 0x09] and dt2 in [0x01, 0x02]:
            # Constant width strings
            if ([dt3, dt4] == [0x00, 0x00] and dt5 > 0) or (0x01 <= dt3 <= 0x07 and dt4 == 0x00):
                width = dt5
                strings = []
                
                # For compressed data, strings are at the end
                if is_compressed:
                    buffer.seek(-width * info.nrows, 2)  # From end
                
                # Read all string data
                string_data = buffer.read(width * info.nrows)
                
                # Extract each string
                for i in range(info.nrows):
                    start = i * width
                    s = string_data[start:start + width].rstrip(b'\0').decode('utf-8', errors='replace')
                    strings.append(s)
                
                return pd.Series(strings)
            
            # Variable width strings
            if [dt3, dt4, dt5] == [0x00, 0x00, 0x00]:
                if dt1 == 0x09:  # Compressed
                    # In JMP file format, compressed variable width strings have:
                    # - Width bytes (1 byte)
                    # - Some header data
                    # - Array of lengths (depends on width_bytes)
                    # - String data
                    
                    # Navigate through the compressed data structure
                    if is_compressed:
                        # First byte after some header gives width of length bytes
                        buffer.seek(9)  # Skip to width_bytes
                        width_bytes = buffer.read(1)[0]
                        buffer.seek(13)  # Skip to string length data
                        
                        if width_bytes == 1:
                            # Read Int8 lengths
                            lengths_data = buffer.read(info.nrows)
                            lengths = np.frombuffer(lengths_data, dtype=np.int8)
                            string_data = buffer.read()  # Rest of the buffer is string data
                        elif width_bytes == 2:
                            # Read Int16 lengths
                            lengths_data = buffer.read(info.nrows * 2)
                            lengths = np.frombuffer(lengths_data, dtype=np.int16)
                            string_data = buffer.read()  # Rest of the buffer is string data
                        else:
                            raise ValueError(f"Unknown width_bytes {width_bytes} in column {column_name}")
                    else:
                        # Uncompressed case needs special handling
                        # This is a complex format with offsets, need to implement based on file structure
                        raise NotImplementedError("Uncompressed variable width strings not yet implemented")
                    
                    # Extract strings based on lengths
                    strings = []
                    pos = 0
                    for length in lengths:
                        if pos + length > len(string_data):
                            # Handle truncated data
                            s = string_data[pos:].decode('utf-8', errors='replace')
                            strings.append(s)
                            break
                            
                        s = string_data[pos:pos + length].decode('utf-8', errors='replace')
                        strings.append(s)
                        pos += length
                    
                    # If we got fewer strings than rows, pad with empty strings
                    while len(strings) < info.nrows:
                        strings.append('')
                    
                    return pd.Series(strings)
    
        # Formula column (usually stored as string)
        if dt1 == 0x08:
            return pd.Series([''] * info.nrows)  # Placeholder
    
    except Exception as e:
        # Log error but don't crash
        print(f"Error parsing column '{column_name}' (idx={column_idx}): {str(e)}")
    
    # If we get here, we don't know how to handle this column type
    print(f"Unknown data type combination: dt=({dt1:02x},{dt2:02x},{dt3:02x},{dt4:02x},{dt5:02x},{dt6:02x}) "
          f"for column {column_name} (idx={column_idx})")
    return pd.Series([float('nan')] * info.nrows)