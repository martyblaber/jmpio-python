"""
Simple example demonstrating how to create a JMP file from Python data
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# Add package to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import write_jmp, read_jmp

# Create a sample DataFrame with various data types
df = pd.DataFrame({
    # Numeric data
    'int8': pd.Series([1, 2, 3, 4, None], dtype='Int8'),
    'int16': pd.Series([-1000, 0, 1000, -5000, None], dtype='Int16'),
    'int32': pd.Series([100000, 200000, -300000, 400000, None], dtype='Int32'),
    'float': [1.1, 2.2, 3.3, 4.4, None],
    
    # String data
    'string_fixed': ['a', 'a', 'a', 'a', None],  # Fixed width
    'string_variable': ['a', 'bb', 'ccc', 'dddd', None],  # Variable width
    
    # Date and time data
    'date': [
        date(2023, 1, 1), 
        date(2023, 2, 15), 
        date(2023, 12, 31), 
        date(2024, 6, 15), 
        None
    ],
    'datetime': [
        datetime(2023, 1, 1, 12, 0, 0), 
        datetime(2023, 2, 15, 13, 30, 45), 
        datetime(2023, 12, 31, 23, 59, 59),
        datetime(2024, 6, 15, 8, 15, 30), 
        None
    ],
    
    # Duration data
    'duration': [
        timedelta(hours=1), 
        timedelta(days=1), 
        timedelta(weeks=1),
        timedelta(minutes=30), 
        None
    ]
})

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Write to JMP file
output_file = 'output/simple_example.jmp'
write_jmp(df, output_file)

print(f"Created JMP file: {output_file}")

# Read back the file to verify it was written correctly
df_read = read_jmp(output_file)

# Compare original and read data
print("\nOriginal DataFrame:")
print(df)
print("\nDataFrame read from JMP file:")
print(df_read)

# Verify column types
print("\nColumn data types:")
for column in df_read.columns:
    print(f"- {column}: {df_read[column].dtype}")

# Check if NaN values were preserved
print("\nNaN values preserved:")
for column in df_read.columns:
    print(f"- {column}: {df_read[column].isna().sum()} NaN values")