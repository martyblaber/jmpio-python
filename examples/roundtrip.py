"""
Example demonstrating roundtrip of JMP files - read existing JMP file and write to a new one
"""

import os
import sys
import pandas as pd

# Add package to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import read_jmp, write_jmp

def main():
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Check if test files are available
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'JMPReader.jl', 'test'))
    
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found.")
        return
    
    # Find all JMP files in the test directory
    jmp_files = [f for f in os.listdir(test_dir) if f.endswith('.jmp')]
    
    if not jmp_files:
        print("No JMP files found in test directory.")
        return
    
    print(f"Found {len(jmp_files)} JMP files to process.")
    
    # Process each JMP file
    for jmp_file in jmp_files:
        input_path = os.path.join(test_dir, jmp_file)
        output_path = os.path.join('output', f"copy_{jmp_file}")
        
        print(f"\nProcessing: {jmp_file}")
        
        try:
            # Read the JMP file
            df = read_jmp(input_path)
            print(f"  Read {len(df)} rows and {len(df.columns)} columns")
            
            # Write to a new file
            write_jmp(df, output_path)
            print(f"  Wrote to {output_path}")
            
            # Read back the file to verify
            df_verify = read_jmp(output_path)
            print(f"  Verified: {len(df_verify)} rows and {len(df_verify.columns)} columns")
            
            # Check if column names match
            orig_cols = set(df.columns)
            new_cols = set(df_verify.columns)
            
            if orig_cols == new_cols:
                print("  Column names match ✓")
            else:
                print("  Column names differ:")
                print(f"    Original only: {orig_cols - new_cols}")
                print(f"    New only: {new_cols - orig_cols}")
            
            # Compare column types
            print("  Column types:")
            for col in orig_cols.intersection(new_cols):
                orig_type = str(df[col].dtype)
                new_type = str(df_verify[col].dtype)
                match = "✓" if orig_type == new_type else "✗"
                print(f"    {col}: {orig_type} -> {new_type} {match}")
            
        except Exception as e:
            print(f"  Error processing {jmp_file}: {str(e)}")


if __name__ == "__main__":
    main()