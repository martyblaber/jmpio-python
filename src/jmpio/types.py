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
    # Additional raw metadata fields preserved for future use
    foo1: np.ndarray | None = None
    charset: str | None = None
    foo2: np.ndarray | None = None
    foo3: np.uint16 | None = None
    # Fields used to help reconstruct file structure
    n_visible: int | None = None
    n_hidden: int | None = None
    idx_visible: np.ndarray | None = None
    idx_hidden: np.ndarray | None = None
    unknown_u32_7: np.ndarray | None = None

    def __str__(self) -> str:
        """
        Human-friendly summary of JMP file metadata.
        Truncates long lists to keep output readable.
        """

        def fmt_list(lst, max_items: int = 10) -> str:
            if lst is None:
                return "None"
            # Normalize numpy arrays for preview only
            if isinstance(lst, np.ndarray):
                seq = lst.tolist()
            else:
                seq = lst
            try:
                n = len(seq)
            except Exception:
                return repr(seq)
            if n <= max_items:
                return str(seq)
            head = ", ".join(map(str, seq[: max_items // 2]))
            tail = ", ".join(map(str, seq[-(max_items // 2) :]))
            return f"[{head}, ..., {tail}] (len={n})"

        names_preview = fmt_list(self.column.names if self.column else None, 6)
        widths_preview = fmt_list(self.column.widths if self.column else None, 8)
        offsets_preview = fmt_list(self.column.offsets if self.column else None, 8)
        idx_vis_preview = fmt_list(self.idx_visible, 12)
        idx_hid_preview = fmt_list(self.idx_hidden, 12)

        return (
            "JMPInfo(\n"
            f"  version={self.version!r}, buildstring={self.buildstring!r}, savetime={self.savetime},\n"
            f"  nrows={self.nrows}, ncols={self.ncols},\n"
            f"  charset={self.charset!r}, foo1={fmt_list(self.foo1)}, foo2={fmt_list(self.foo2)}, foo3={self.foo3},\n"
            f"  n_visible={self.n_visible}, n_hidden={self.n_hidden},\n"
            f"  idx_visible={idx_vis_preview}, idx_hidden={idx_hid_preview},\n"
            f"  unknown_u32_7={fmt_list(self.unknown_u32_7)},\n"
            f"  column.names={names_preview},\n"
            f"  column.widths={widths_preview},\n"
            f"  column.offsets={offsets_preview}\n"
            ")"
        )


@dataclass
class RowState:
    """Row state information (marker and color)"""

    marker: str
    color: tuple[float, float, float]  # RGB values (0-1)


# Type mappings from JMP to Python
JMP_TYPE_MAP = {
    "float": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "string": str,
    "date": "datetime64[D]",
    "datetime": "datetime64[s]",
    "time": "datetime64[s]",
    "duration": "timedelta64[ms]",
}
