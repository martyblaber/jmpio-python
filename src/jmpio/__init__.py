"""
jmpio - A package for reading and writing SAS JMP files

This package provides functionality to read and write binary JMP files from
the SAS JMP statistical software.
"""

from .reader import read_jmp, scan_directory
from .writer import write_jmp

__all__ = ["read_jmp", "scan_directory", "write_jmp"]
__version__ = "0.1.0"
