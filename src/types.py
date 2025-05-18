"""
Data types and structures for JMP files
"""

from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class Column:
    """Column information for a JMP file"""
    names: list[str]
    widths: list[int]
    offsets: list[int]


@dataclass
class JMPInfo:
    """Metadata information for a JMP file"""
    version: str
    buildstring: str
    savetime: datetime
    nrows: int
    ncols: int
    column: Column


@dataclass
class RowState:
    """Row state information (marker and color)"""
    marker: str
    color: tuple[float, float, float]  # RGB values (0-1)


# Type mappings from JMP to Python
JMP_TYPE_MAP = {
    'float': np.float64,
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'string': str,
    'date': 'datetime64[D]',
    'datetime': 'datetime64[s]',
    'time': 'datetime64[s]',
    'duration': 'timedelta64[ms]',
}