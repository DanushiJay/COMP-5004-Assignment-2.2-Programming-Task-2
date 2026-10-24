import numpy as np
import pytest

from numcompute import io as nc_io


def write(tmp_path, text, name="d.csv"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_load_csv_fills_missing_with_nan(tmp_path):
    p = write(tmp_path, "a,b,c\n1,2,3\n4,,6\n,8,9\n")
    X = nc_io.load_csv(p, skip_header=1)
    assert X.shape == (3, 3) and X.dtype == float
    assert np.isnan(X[1, 1]) and np.isnan(X[2, 0])
    assert X[0, 2] == 3.0


def test_load_csv_custom_fill_and_na_tokens(tmp_path):
    p = write(tmp_path, "1,NA\n2,5\n")
    out = nc_io.load_csv(p, fill_value=-1.0)
    np.testing.assert_array_equal(out, [[1, -1], [2, 5]])


def test_load_csv_skip_missing_rows(tmp_path):
    p = write(tmp_path, "1,2\n3,\n5,6\n")
    out = nc_io.load_csv(p, missing="skip")
    np.testing.assert_array_equal(out, [[1, 2], [5, 6]])


def test_load_csv_tab_delimiter(tmp_path):
    p = write(tmp_path, "1\t2\n3\t4\n")
    out = nc_io.load_csv(p, delimiter="\t")
    np.testing.assert_array_equal(out, [[1, 2], [3, 4]])


def test_load_csv_ignores_blank_lines_and_comments(tmp_path):
    p = write(tmp_path, "# note\n1,2\n\n3,4\n")
    np.testing.assert_array_equal(nc_io.load_csv(p), [[1, 2], [3, 4]])


def test_load_csv_single_column_and_single_row_stay_2d(tmp_path):
    col = write(tmp_path, "1\n2\n3\n", "col.csv")
    row = write(tmp_path, "1,2,3\n", "row.csv")
    one = write(tmp_path, "7\n", "one.csv")
    assert nc_io.load_csv(col).shape == (3, 1)
    assert nc_io.load_csv(row).shape == (1, 3)
    assert nc_io.load_csv(one).shape == (1, 1)


def test_load_csv_empty_and_header_only_files_raise(tmp_path):
    with pytest.raises(ValueError, match="no data rows"):
        nc_io.load_csv(write(tmp_path, "", "e.csv"))
    with pytest.raises(ValueError, match="no data rows"):
        nc_io.load_csv(write(tmp_path, "a,b\n", "h.csv"), skip_header=1)


def test_load_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        nc_io.load_csv(str(tmp_path / "nope.csv"))


def test_load_csv_inconsistent_columns_raises(tmp_path):
    p = write(tmp_path, "1,2,3\n4,5\n6,7,8\n")
    with pytest.raises(ValueError, match="Malformed CSV"):
        nc_io.load_csv(p)


def test_load_csv_unreadable_text_raises_instead_of_becoming_nan(tmp_path):
    p = write(tmp_path, "1,2\n3,abc\n")
    with pytest.raises(ValueError, match="Non-numeric"):
        nc_io.load_csv(p)


def test_load_csv_int_dtype_rules(tmp_path):
    p = write(tmp_path, "1,2\n3,\n")
    with pytest.raises(ValueError, match="cannot hold NaN"):
        nc_io.load_csv(p, dtype=int)
    out = nc_io.load_csv(p, dtype=int, missing="skip")
    assert out.dtype.kind == "i" and out.tolist() == [[1, 2]]
    filled = nc_io.load_csv(p, dtype=int, fill_value=0)
    assert filled.tolist() == [[1, 2], [3, 0]]


def test_load_csv_int_dtype_rejects_fractions(tmp_path):
    p = write(tmp_path, "1,2.5\n")
    with pytest.raises(ValueError, match="Non-integer"):
        nc_io.load_csv(p, dtype=int, fill_value=0)


def test_load_csv_string_dtype_keeps_text_and_skip_drops_missing(tmp_path):
    p = write(tmp_path, "a,NA\nb,x\n,y\n")
    table = nc_io.load_csv(p, dtype=str)
    assert table.shape == (3, 2) and table[0, 1] == "NA"
    kept = nc_io.load_csv(p, dtype=str, missing="skip")
    assert kept.tolist() == [["b", "x"]]


def test_load_csv_invalid_missing_option(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        nc_io.load_csv(write(tmp_path, "1\n"), missing="drop")


def test_chunks_concatenate_to_full_read(tmp_path):
    rows = "\n".join(f"{i},{i * 2.5}" for i in range(23))
    p = write(tmp_path, "x,y\n" + rows + "\n")
    chunks = list(nc_io.iter_csv_chunks(p, chunk_size=5, skip_header=1))
    assert [c.shape[0] for c in chunks] == [5, 5, 5, 5, 3]
    full = nc_io.load_csv(p, skip_header=1)
    np.testing.assert_array_equal(np.vstack(chunks), full)


def test_chunks_with_missing_values_and_skip(tmp_path):
    p = write(tmp_path, "1,2\n3,\n5,6\n7,8\n")
    full = nc_io.load_csv(p, missing="skip")
    chunks = nc_io.iter_csv_chunks(p, chunk_size=2, missing="skip")
    np.testing.assert_array_equal(np.vstack(list(chunks)), full)
    filled = np.vstack(list(nc_io.iter_csv_chunks(p, chunk_size=3)))
    np.testing.assert_array_equal(filled, nc_io.load_csv(p))


def test_chunks_single_column_and_chunk_larger_than_file(tmp_path):
    p = write(tmp_path, "1\n2\n3\n")
    chunks = list(nc_io.iter_csv_chunks(p, chunk_size=100))
    assert len(chunks) == 1 and chunks[0].shape == (3, 1)


def test_chunks_later_chunk_with_different_width_raises(tmp_path):
    p = write(tmp_path, "1,2\n3,4\n5,6,7\n8,9,10\n")
    with pytest.raises(ValueError, match="Malformed CSV"):
        list(nc_io.iter_csv_chunks(p, chunk_size=2))


def test_chunks_single_row_chunk_with_wrong_width_raises(tmp_path):
    p = write(tmp_path, "1\n2\n3,4,5\n")
    with pytest.raises(ValueError, match="Malformed CSV"):
        list(nc_io.iter_csv_chunks(p, chunk_size=2))


def test_chunks_invalid_args_and_empty_file(tmp_path):
    p = write(tmp_path, "1,2\n")
    for bad in (0, -3, 1.5):
        with pytest.raises(ValueError, match="chunk_size"):
            list(nc_io.iter_csv_chunks(p, chunk_size=bad))
    with pytest.raises(ValueError, match="no data rows"):
        list(nc_io.iter_csv_chunks(write(tmp_path, "", "e.csv")))


def test_read_header(tmp_path):
    p = write(tmp_path, "age, city ,score\n1,2,3\n")
    assert nc_io.read_header(p) == ["age", "city", "score"]
    with pytest.raises(ValueError):
        nc_io.read_header(p, line=5)


def test_mixed_csv_split_feeds_scaler_and_encoder(tmp_path):
    from numcompute.preprocessing import (
        OneHotEncoder, SimpleImputer, StandardScaler,
    )
    text = "age,city,score\n30,Adl,1.5\n,Syd,2.5\n41,Adl,NA\n25,Syd,3\n"
    p = write(tmp_path, text)
    X_num, X_cat = nc_io.load_mixed_csv(p, [0, 2], [1], skip_header=1)
    assert X_num.shape == (4, 2) and X_cat.shape == (4, 1)
    assert np.isnan(X_num[1, 0]) and np.isnan(X_num[2, 1])
    filled = SimpleImputer("mean").fit_transform(X_num)
    Z = StandardScaler().fit_transform(filled)
    assert not np.isnan(Z).any()
    assert OneHotEncoder().fit_transform(X_cat).shape == (4, 2)


def test_split_columns_errors():
    table = np.array([["1", "a"], ["x", "b"]])
    with pytest.raises(ValueError, match="non-numeric"):
        nc_io.split_columns(table, [0], [1])
    with pytest.raises(ValueError, match="out of range"):
        nc_io.split_columns(table, [5], [1])
    with pytest.raises(ValueError, match="2-D"):
        nc_io.split_columns(np.array(["a"]), [0], [])


def test_split_columns_accepts_non_contiguous_table():
    table = np.array([["1", "a", "2", "b"],
                      ["3", "c", "4", "d"]])[:, ::2]  # keeps columns 0, 2
    assert not table.flags["C_CONTIGUOUS"]
    X_num, X_cat = nc_io.split_columns(table, [0, 1], [])
    np.testing.assert_array_equal(X_num, [[1.0, 2.0], [3.0, 4.0]])
    assert X_cat.shape == (2, 0)
