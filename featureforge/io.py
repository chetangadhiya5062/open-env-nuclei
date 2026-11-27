"""Reading and writing tables: CSV / TSV, Excel, Parquet, JSON / JSON Lines, Feather.

``read_table`` picks the reader from the file extension (or the ``fmt`` argument), retries common text encodings for
CSV, sniffs the delimiter, and raises ``TableReadError`` with a message a user can act on.
"""

import csv
import hashlib
import io
import logging
from pathlib import Path
from typing import BinaryIO, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_READ = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "csv",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".feather": "feather",
}
SUPPORTED_WRITE = ("csv", "parquet", "json", "jsonl", "excel", "feather")
_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

Source = Union[str, Path, bytes, BinaryIO]


class TableReadError(ValueError):
    """The file could not be read as a table; the message explains why."""


def detect_format(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".xls":
        raise TableReadError("legacy .xls files are not supported; save the sheet as .xlsx or .csv")
    if suffix not in SUPPORTED_READ:
        raise TableReadError(
            f"unsupported file type {suffix or '(none)'!r}; supported: {', '.join(sorted(SUPPORTED_READ))}"
        )
    return SUPPORTED_READ[suffix]


def _to_bytes(src: Source) -> bytes:
    if isinstance(src, (bytes, bytearray)):
        return bytes(src)
    if isinstance(src, (str, Path)):
        return Path(src).read_bytes()
    return src.read()


def _read_csv(data: bytes, max_rows: Optional[int]) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for enc in _ENCODINGS:
        try:
            text = data.decode(enc)
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
        head = text[:20_000]
        try:
            sep = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
        except csv.Error:
            sep = ","
        try:
            return pd.read_csv(io.StringIO(text), sep=sep, nrows=max_rows, low_memory=False)
        except pd.errors.EmptyDataError as exc:
            raise TableReadError("the file is empty") from exc
        except pd.errors.ParserError as exc:
            raise TableReadError(f"could not parse the CSV file: {exc}") from exc
    raise TableReadError(f"could not decode the file as text ({last_exc})")


def read_table(
    src: Source, name: Optional[str] = None, fmt: Optional[str] = None, max_rows: Optional[int] = None
) -> pd.DataFrame:
    """Read a table. ``name`` (a file name) is needed to detect the format when ``src`` is bytes or a stream."""
    if fmt is None:
        if name is None and isinstance(src, (str, Path)):
            name = str(src)
        if name is None:
            raise TableReadError("cannot detect the file format: pass a file name or fmt=")
        fmt = detect_format(name)
    data = _to_bytes(src)
    if not data:
        raise TableReadError("the file is empty")
    try:
        if fmt == "csv":
            df = _read_csv(data, max_rows)
        elif fmt == "excel":
            df = pd.read_excel(io.BytesIO(data), nrows=max_rows, engine="openpyxl")
        elif fmt == "parquet":
            df = pd.read_parquet(io.BytesIO(data))
        elif fmt == "feather":
            df = pd.read_feather(io.BytesIO(data))
        elif fmt == "jsonl":
            df = pd.read_json(io.BytesIO(data), lines=True, nrows=max_rows)
        elif fmt == "json":
            try:
                df = pd.read_json(io.BytesIO(data))
            except ValueError:
                df = pd.read_json(io.BytesIO(data), lines=True)
        else:
            raise TableReadError(f"unknown format {fmt!r}")
    except TableReadError:
        raise
    except Exception as exc:  # pyarrow / openpyxl / json errors -> one readable error type
        raise TableReadError(f"could not read the file as {fmt}: {type(exc).__name__}: {exc}") from exc
    if max_rows is not None:
        df = df.head(max_rows)
    if df.shape[1] == 0:
        raise TableReadError("the file has no columns")
    df.columns = _unique_names([str(c) for c in df.columns])
    logger.info("read %s table: %d rows x %d columns", fmt, len(df), df.shape[1])
    return df


def _unique_names(names: list) -> list:
    """Make column names unique and non-empty (``a, a`` -> ``a, a_1``)."""
    seen: dict = {}
    out = []
    for i, n in enumerate(names):
        n = n.strip() or f"column_{i}"
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        seen.setdefault(n, 0)
        out.append(n)
    return out


def write_table(df: pd.DataFrame, path: Union[str, Path], fmt: Optional[str] = None) -> Path:
    """Write ``df``; the format comes from ``fmt`` or the file extension."""
    path = Path(path)
    fmt = fmt or {".xlsx": "excel", ".pq": "parquet", ".ndjson": "jsonl"}.get(path.suffix.lower(), path.suffix[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        _parquet_safe(df).to_parquet(path, index=False)
    elif fmt == "feather":
        _parquet_safe(df).reset_index(drop=True).to_feather(path)
    elif fmt == "json":
        df.to_json(path, orient="records", date_format="iso", indent=1)
    elif fmt == "jsonl":
        df.to_json(path, orient="records", lines=True, date_format="iso")
    elif fmt == "excel":
        df.to_excel(path, index=False, engine="openpyxl")
    else:
        raise ValueError(f"unsupported output format {fmt!r}; choose one of {SUPPORTED_WRITE}")
    return path


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Arrow cannot store object columns with mixed Python types; turn those into strings."""
    out = df
    for c in df.columns:
        if df[c].dtype == object:
            kinds = {type(v) for v in df[c].dropna().head(5000)}
            if len(kinds) > 1:
                out = out.assign(**{c: df[c].map(lambda v: v if pd.isna(v) else str(v))})
    return out


def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
