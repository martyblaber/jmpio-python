# jmpio

A Python package for reading and writing SAS JMP files.

## Description

`jmpio` is a mostly-faithful Python port of the Julia package [JMPReader.jl](https://github.com/jaakkor2/JMPReader.jl) by [@jaakkor2](https://github.com/jaakkor2). It aims to be able to read and write binary JMP files from SAS JMP statistical software. There's no way this package would exist without @jaakor2's efforts.

> [!WARNING]  
> `jmpio` cannot actually write a valid jmp file - it's a work in progress.
> `jmpio` is not particularly efficient and uses pandas DataFrames internally. 

## Features

- Read JMP files into pandas DataFrames
- Write pandas DataFrames to JMP files (eventually)
  - Embed Scripts in those files for plotting data.
- Support for all JMP data types:
  - Numeric (Float64, Int8, Int16, Int32)
  - Character (fixed width and variable width strings)
  - Date/Time (Date, DateTime, Time, Duration)
  - Geographic (Latitude/Longitude)
  - Row states (markers and colors)
  - Currency
- Support for compressed and uncompressed JMP files
- Column selection/filtering when reading
- Strong typing throughout the codebase

## Installation

```bash
pip install jmpio
```

## Basic Usage

### Reading a JMP file

```python
import jmpio
import pandas as pd

# Read a JMP file into a pandas DataFrame
df = jmpio.read_jmp("path/to/file.jmp")

# Select specific columns
df = jmpio.read_jmp("path/to/file.jmp", select=["Column1", "Column2"])

# Drop specific columns
df = jmpio.read_jmp("path/to/file.jmp", drop=["Column1", "Column2"])

# Find all JMP files in a directory
file_info = jmpio.scan_directory("path/to/directory")
```

### Writing a JMP file

```python
import jmpio
import pandas as pd
from datetime import datetime, date

# Create a DataFrame
df = pd.DataFrame({
    'integers': [1, 2, 3, 4],
    'floats': [1.1, 2.2, 3.3, 4.4],
    'strings': ['a', 'bb', 'ccc', 'dddd'],
    'dates': [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)]
})

# Write to a JMP file with compression (default)
jmpio.write_jmp(df, "output.jmp")

# Write to a JMP file without compression
jmpio.write_jmp(df, "output_uncompressed.jmp", compress=False)

# Specify a specific JMP version
jmpio.write_jmp(df, "output_v16.jmp", version="16.0")
```

## Supported Data Types

### Reading

The following data types are supported when reading JMP files:

| JMP Data Type | Python Data Type |
|---------------|------------------|
| Numeric       | float64, int8, int16, int32 |
| Character     | str (fixed or variable width) |
| Date          | datetime64[D] / datetime.date |
| Time          | datetime64[s] / datetime.time |
| DateTime      | datetime64[s] / datetime.datetime |
| Duration      | timedelta64[ms] / datetime.timedelta |
| Row State     | jmpio.RowState |
| Geographic    | float64 |
| Currency      | float64 |

### Writing

The following data types are supported when writing to JMP files:

| Python Data Type | JMP Data Type |
|------------------|---------------|
| int, Int8, Int16, Int32 | Integer |
| float, Float64 | Numeric |
| str | Character |
| datetime64, datetime.datetime | DateTime |
| datetime64[D], datetime.date | Date |
| datetime.time | Time |
| timedelta64, datetime.timedelta | Duration |
| jmpio.RowState | Row State |

## Requirements

- Python 3.10+
- NumPy
- pandas

## License

MIT

## Acknowledgments

This package is a Python port of the Julia package [JMPReader.jl](https://github.com/jaakkor2/JMPReader.jl).
JMP is a registered trademark of SAS Institute Inc. This project is not affiliated with, sponsored by, or endorsed by SAS Institute Inc.


