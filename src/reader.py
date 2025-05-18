"""
Main function for reading JMP files
"""

import os
from typing import BinaryIO, List, Optional, Union, Dict, Any, Pattern
import pandas as pd
import re

from .types import JMPInfo
from .utils import check_magic
from .metadata import read_metadata
from .column import read_column_data


def read_jmp(filename: str, 
             select: Optional[List[Union[int, str, Pattern]]] = None, 
             drop: Optional[List[Union[int, str, Pattern]]] = None) -> pd.DataFrame:
    """
    Read a JMP file and return a pandas DataFrame
    
    Parameters:
    -----------
    filename : str
        Path to the JMP file
    select : Optional[List[Union[int, str, Pattern]]], default=None
        List of columns to include. Can contain indices, strings, or regex patterns.
    drop : Optional[List[Union[int, str, Pattern]]], default=None
        List of columns to exclude. Can contain indices, strings, or regex patterns.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the JMP file data
    
    Examples:
    ---------
    >>> import jmpio
    >>> df = jmpio.read_jmp("example.jmp")
    >>> 
    >>> # Select specific columns
    >>> df = jmpio.read_jmp("example.jmp", select=["Column1", "Column2"])
    >>> 
    >>> # Use regex to select columns
    >>> import re
    >>> df = jmpio.read_jmp("example.jmp", select=[re.compile("^char")])
    >>> 
    >>> # Drop columns
    >>> df = jmpio.read_jmp("example.jmp", drop=[0, 1, 2])
    
    Raises:
    -------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file is not a valid JMP file or is corrupted
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File '{filename}' does not exist")
    
    # Open file in binary mode
    with open(filename, 'rb') as file:
        # Verify JMP file signature
        if not check_magic(file):
            raise ValueError(f"'{filename}' is not a valid JMP file or is corrupted")
        
        # Read metadata
        info = read_metadata(file)
        
        # Filter columns based on select/drop parameters
        col_indices = list(range(info.ncols))
        
        if select is not None:
            selected_indices = []
            for item in select:
                if isinstance(item, int):
                    if 0 <= item < info.ncols:
                        selected_indices.append(item)
                elif isinstance(item, str):
                    try:
                        idx = info.column.names.index(item)
                        selected_indices.append(idx)
                    except ValueError:
                        pass  # Column name not found
                elif hasattr(item, 'search'):  # Regex pattern
                    for i, name in enumerate(info.column.names):
                        if item.search(name):
                            selected_indices.append(i)
            col_indices = list(set(selected_indices))
        
        if drop is not None:
            drop_indices = []
            for item in drop:
                if isinstance(item, int):
                    if 0 <= item < info.ncols:
                        drop_indices.append(item)
                elif isinstance(item, str):
                    try:
                        idx = info.column.names.index(item)
                        drop_indices.append(idx)
                    except ValueError:
                        pass  # Column name not found
                elif hasattr(item, 'search'):  # Regex pattern
                    for i, name in enumerate(info.column.names):
                        if item.search(name):
                            drop_indices.append(i)
            col_indices = [i for i in col_indices if i not in drop_indices]
        
        # Sort indices to maintain original column order
        col_indices.sort()
        
        # Read data for each column
        df = pd.DataFrame()
        for i in col_indices:
            try:
                data = read_column_data(file, info, i)
                df[info.column.names[i]] = data
            except Exception as e:
                print(f"Error reading column '{info.column.names[i]}': {str(e)}")
                # Add a column of NaN values as a placeholder
                df[info.column.names[i]] = [float('nan')] * info.nrows
        
        # Make sure the DataFrame has the correct number of rows
        if len(df) != info.nrows and len(df) > 0:
            # Adjust if necessary (can happen with string columns)
            for col in df.columns:
                if len(df[col]) < info.nrows:
                    # Pad with appropriate missing values
                    pad_len = info.nrows - len(df[col])
                    col_dtype = df[col].dtype
                    
                    if pd.api.types.is_numeric_dtype(col_dtype):
                        df[col] = pd.concat([df[col], pd.Series([float('nan')] * pad_len)])
                    elif pd.api.types.is_string_dtype(col_dtype):
                        df[col] = pd.concat([df[col], pd.Series([''] * pad_len)])
                    else:
                        df[col] = pd.concat([df[col], pd.Series([None] * pad_len)])
        
        return df


def scan_directory(directory: str, recursive: bool = True) -> pd.DataFrame:
    """
    Scan a directory for JMP files and read them
    
    Parameters:
    -----------
    directory : str
        Directory to scan
    recursive : bool, default=True
        Whether to scan subdirectories recursively
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with information about each JMP file:
        - filename: Path to the JMP file
        - version: JMP version
        - rows: Number of rows
        - columns: Number of columns
        - savetime: When the file was saved
    """
    import os
    
    results = []
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jmp'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        if check_magic(f):
                            info = read_metadata(f)
                            results.append({
                                'filename': file_path,
                                'version': info.version,
                                'rows': info.nrows,
                                'columns': info.ncols,
                                'savetime': info.savetime
                            })
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
        
        # If not recursive, break after first level
        if not recursive:
            break
    
    # Convert to DataFrame
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=['filename', 'version', 'rows', 'columns', 'savetime'])