import os
from itertools import islice
import numpy as np

__all__ = [
    "DEFAULT_MISSING",
    "load_csv",
    "iter_csv_chunks",
    "read_header",
    "split_columns",
    "load_mixed_csv",
]

DEFAULT_MISSING = ("", "NA", "N/A", "NaN", "nan", "NULL", "null")


def _is_str(dtype):
    return np.dtype(dtype).kind in "USO"


def _check_options(missing, dtype, fill_value):
    if missing not in ("fill", "skip"):
        raise ValueError(f"missing must be 'fill' or 'skip', got {missing!r}")
    int_like = not _is_str(dtype) and np.dtype(dtype).kind in "iub"
    nan_fill = isinstance(fill_value, float) and np.isnan(fill_value)
    if int_like and missing == "fill" and nan_fill:
        raise ValueError(
            f"dtype {np.dtype(dtype)} cannot hold NaN: pass a numeric "
            "fill_value or use missing='skip'"
        )


def _open(path, encoding):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path!r}")
    return open(path, "r", encoding=encoding, newline=None)


def _data_lines(lines):
    """Keep lines that are neither blank nor comments."""
    return [ln for ln in lines
            if ln.strip() and not ln.lstrip().startswith("#")]


def _n_columns(line, delimiter):
    return len(line.strip().split(delimiter))


def _read_table(lines, delimiter, ncols):
    """Split lines into a string table of shape (len(lines), ncols).

    ``genfromtxt`` squeezes a single row or column to 1-D, so the element
    count is checked against ``len(lines) * ncols``. That also catches a
    later chunk whose column count differs from the first chunk.
    """
    try:
        arr = np.genfromtxt(
            lines, delimiter=delimiter, dtype=str,
            autostrip=True, invalid_raise=True,
        )
    except ValueError as exc:
        raise ValueError(
            f"Malformed CSV (expected {ncols} columns per row): {exc}"
        ) from exc
    arr = np.atleast_1d(arr)
    wrong_size = arr.size != len(lines) * ncols
    wrong_width = arr.ndim == 2 and arr.shape[1] != ncols
    if wrong_size or wrong_width:
        raise ValueError(
            f"Malformed CSV: expected {ncols} columns per row in "
            f"{len(lines)} rows, got {arr.size} values"
        )
    return arr.reshape(-1, ncols)


def _convert(table, dtype, missing, fill_value, missing_values):
    """Apply missing-value handling and dtype conversion to a string table."""
    miss = np.isin(table, list(missing_values))
    if missing == "skip":
        keep = ~miss.any(axis=1)
        table, miss = table[keep], miss[keep]
    if _is_str(dtype):
        return table
    try:
        out = np.where(miss, "nan", table).astype(float)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric text in numeric CSV data: {exc}"
        ) from exc
    if missing == "fill":
        out[miss] = fill_value
    if np.dtype(dtype).kind in "iu" and not np.all(out == np.floor(out)):
        raise ValueError(f"Non-integer values cannot be cast to {dtype}")
    return out if np.dtype(dtype) == np.float64 else out.astype(dtype)


def _parse(lines, delimiter, dtype, missing, fill_value, ncols,
           missing_values):
    table = _read_table(lines, delimiter, ncols)
    return _convert(table, dtype, missing, fill_value, missing_values)


def _skip_header_lines(f, skip_header):
    for _ in range(skip_header):
        next(f, None)
    return f


def load_csv(path, delimiter=",", skip_header=0, missing="fill",
             fill_value=np.nan, dtype=float, missing_values=DEFAULT_MISSING,
             encoding="utf-8"):
    """Read a delimited text file fully into a 2-D NumPy array.

    Parameters
    ----------
    path : str or os.PathLike
        File to read.
    delimiter : str, default ","
        Field separator (e.g. ``"\\t"`` for TSV).
    skip_header : int, default 0
        Number of leading lines to ignore (e.g. 1 for a header row).
    missing : {"fill", "skip"}, default "fill"
        ``"fill"`` replaces missing numeric fields by ``fill_value``;
        ``"skip"`` drops every row that contains a missing field.
    fill_value : scalar, default nan
        Placeholder for missing fields (string dtypes keep the text).
    dtype : dtype, default float
        Output dtype. Numbers are parsed as float and then cast, so a
        missing field is never silently turned into a sentinel like -1.
    missing_values : iterable of str
        Field texts treated as missing.
    encoding : str, default "utf-8"

    Returns
    -------
    ndarray, shape (n_rows, n_cols); always 2-D.

    Shapes
    ------
    file with r data rows and c columns -> (r, c); a single-column file
    gives (r, 1); with ``missing="skip"`` fewer rows may remain.

    Raises
    ------
    FileNotFoundError : path does not exist.
    ValueError : empty file or no data rows; rows with different column
        counts; unknown ``missing``; an integer dtype with NaN fill;
        unreadable text in a numeric column.

    Notes
    -----
    Time O(file size); space O(file size) (use ``iter_csv_chunks`` for
    files larger than memory).
    """
    _check_options(missing, dtype, fill_value)
    with _open(path, encoding) as f:
        lines = _data_lines(_skip_header_lines(f, skip_header).readlines())
    if not lines:
        raise ValueError(f"CSV file {path!r} contains no data rows")
    ncols = _n_columns(lines[0], delimiter)
    return _parse(lines, delimiter, dtype, missing, fill_value, ncols,
                  missing_values)


def iter_csv_chunks(path, chunk_size=10_000, delimiter=",", skip_header=0,
                    missing="fill", fill_value=np.nan, dtype=float,
                    missing_values=DEFAULT_MISSING, encoding="utf-8"):
    """Stream a CSV file as 2-D arrays of at most ``chunk_size`` rows.

    Only one chunk is held in memory at a time. Parameters are as in
    ``load_csv``. With ``missing="skip"`` a chunk can contain fewer than
    ``chunk_size`` rows (an empty chunk is not yielded).

    Parameters
    ----------
    path : str or os.PathLike
    chunk_size : int, default 10_000
        Number of lines read per chunk; must be >= 1.

    Yields
    ------
    ndarray, shape (<=chunk_size, n_cols)

    Shapes
    ------
    concatenating all chunks along axis 0 equals ``load_csv`` with the
    same arguments. The column count is fixed by the first data line.

    Raises
    ------
    FileNotFoundError, ValueError : as in ``load_csv``. A malformed-row
        message refers to the offending chunk, not the whole file.

    Notes
    -----
    Time O(file size); space O(chunk_size * n_cols). The while-loop is one
    of the sanctioned loops: it only iterates over blocks of the file.
    """
    bad_type = isinstance(chunk_size, bool) or not isinstance(
        chunk_size, (int, np.integer))
    if bad_type or chunk_size < 1:
        raise ValueError(
            f"chunk_size must be a positive integer, got {chunk_size!r}"
        )
    _check_options(missing, dtype, fill_value)
    with _open(path, encoding) as f:
        _skip_header_lines(f, skip_header)
        ncols = None
        while True:
            lines = _data_lines(list(islice(f, chunk_size)))
            if not lines:
                break
            if ncols is None:
                ncols = _n_columns(lines[0], delimiter)
            chunk = _parse(lines, delimiter, dtype, missing, fill_value,
                           ncols, missing_values)
            if chunk.shape[0]:
                yield chunk
        if ncols is None:
            raise ValueError(f"CSV file {path!r} contains no data rows")


def read_header(path, delimiter=",", line=0, encoding="utf-8"):
    """Return the column names found on line ``line`` of the file.

    Parameters
    ----------
    path : str or os.PathLike
    delimiter : str, default ","
    line : int, default 0
        Zero-based line holding the names.
    encoding : str, default "utf-8"

    Returns
    -------
    list of str, one per column (whitespace stripped).

    Shapes
    ------
    file with c columns -> list of length c

    Raises
    ------
    FileNotFoundError : missing file.
    ValueError : the file has fewer than ``line + 1`` lines.

    Notes
    -----
    Time O(line + length of the header line), space O(c).
    """
    with _open(path, encoding) as f:
        row = next(islice(f, line, line + 1), None)
    if row is None:
        raise ValueError(f"CSV file {path!r} has no line {line}")
    return [name.strip() for name in row.strip().split(delimiter)]


def split_columns(table, numeric_cols, categorical_cols,
                  missing_values=DEFAULT_MISSING):
    """Split a string table into a float block and a category block.

    Parameters
    ----------
    table : ndarray of str, shape (n, d)
        Typically from ``load_csv(..., dtype=str)``.
    numeric_cols, categorical_cols : sequence of int
        Column indices of each kind (may be empty).
    missing_values : iterable of str
        Texts converted to NaN in the numeric block. The categorical block
        keeps its text, so a missing category becomes its own category.

    Returns
    -------
    (X_num, X_cat) : float ndarray (n, k), str ndarray (n, m)

    Shapes
    ------
    in: (n, d) -> out: (n, len(numeric_cols)), (n, len(categorical_cols))

    Raises
    ------
    ValueError : table not 2-D, a column index out of range, or a numeric
        column holding text that is not a number.

    Notes
    -----
    Time O(n*(k+m)), space O(n*(k+m)).
    """
    table = np.asarray(table)
    if table.ndim != 2:
        raise ValueError(f"table must be 2-D, got shape {table.shape}")
    d = table.shape[1]
    num = np.asarray(numeric_cols, dtype=int).reshape(-1)
    cat = np.asarray(categorical_cols, dtype=int).reshape(-1)
    idx = np.concatenate((num, cat))
    if idx.size and (idx.min() < -d or idx.max() >= d):
        raise ValueError(
            f"column index out of range for table with {d} columns"
        )
    text = np.char.strip(table[:, num].astype(str))
    text = np.where(np.isin(text, list(missing_values)), "nan", text)
    try:
        X_num = text.astype(float)
    except ValueError as exc:
        raise ValueError(
            f"numeric column contains non-numeric text: {exc}"
        ) from exc
    return X_num, table[:, cat].astype(str)


def load_mixed_csv(path, numeric_cols, categorical_cols, delimiter=",",
                   skip_header=0, missing_values=DEFAULT_MISSING,
                   encoding="utf-8"):
    """Read a mixed CSV and return ``(X_num, X_cat)``.

    Parameters
    ----------
    path : str or os.PathLike
    numeric_cols, categorical_cols : sequence of int
        Column indices of each kind.
    delimiter, skip_header, missing_values, encoding :
        As in ``load_csv``.

    Returns
    -------
    (X_num, X_cat) : float ndarray (n, k) with NaN for missing, and a
        str ndarray (n, m).

    Shapes
    ------
    file (n rows, d cols) -> (n, k), (n, m)

    Raises
    ------
    FileNotFoundError, ValueError : as in ``load_csv`` and
        ``split_columns``.

    Notes
    -----
    Time and space O(file size).
    """
    table = load_csv(path, delimiter=delimiter, skip_header=skip_header,
                     dtype=str, missing_values=missing_values,
                     encoding=encoding)
    return split_columns(table, numeric_cols, categorical_cols,
                         missing_values)
