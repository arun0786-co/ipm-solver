import gzip

import numpy as np
import pytest

from ipm.mps import read_mps

FREE = """NAME          TESTLP
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  MYEQN
 E  R4
COLUMNS
    X1        COST         1.0   LIM1         1.0
    X1        LIM2         1.0
    X2        COST         2.0   LIM1         1.0
    X2        MYEQN       -1.0
    X3        COST        -1.0   MYEQN        1.0
    X3        R4           2.0
    X4        COST         1.0   R4           1.0
RHS
    RHS       COST        -7.5
    RHS       LIM1         4.0   LIM2         1.0
    RHS       MYEQN        7.0   R4           3.0
RANGES
    RNG       LIM1         2.5   R4          -1.0
BOUNDS
 UP BND       X1           4.0
 MI BND       X2
 UP BND       X3          -2.0
 FR BND       X4
ENDATA
"""


def test_free_format(tmp_path):
    p = tmp_path / "t.mps"
    p.write_text(FREE)
    lp = read_mps(p)
    assert lp.shape == (4, 4)
    np.testing.assert_allclose(lp.c, [1, 2, -1, 1])
    assert lp.c0 == 7.5
    # L row with range: [rhs - |R|, rhs]
    assert (lp.rl[0], lp.ru[0]) == (1.5, 4.0)
    assert (lp.rl[1], lp.ru[1]) == (1.0, np.inf)
    assert (lp.rl[2], lp.ru[2]) == (7.0, 7.0)
    # E row with negative range: [rhs + R, rhs]
    assert (lp.rl[3], lp.ru[3]) == (2.0, 3.0)
    # negative UP with zero lower bound makes the lower bound -inf
    np.testing.assert_array_equal(lp.l, [0, -np.inf, -np.inf, -np.inf])
    np.testing.assert_array_equal(lp.u, [4, np.inf, -2, np.inf])
    A = lp.A.toarray()
    np.testing.assert_allclose(A, [[1, 1, 0, 0], [1, 0, 0, 0], [0, -1, 1, 0], [0, 0, 2, 1]])


def fixed_line(*fields):
    """Place fields at the fixed-MPS columns 2, 5, 15, 25, 40, 50."""
    starts = [1, 4, 14, 24, 39, 49]
    line = [" "] * 61
    for start, f in zip(starts, fields):
        line[start:start + len(f)] = f
    return "".join(line).rstrip()


def test_gzip_and_fixed_names_with_spaces(tmp_path):
    # fixed MPS allows spaces inside names, which free format cannot split
    text = "\n".join([
        "NAME          SPACES",
        "ROWS",
        fixed_line("N", "OBJ"),
        fixed_line("G", "ROW A"),
        "COLUMNS",
        fixed_line("", "COL 1", "OBJ", "1.", "ROW A", "1."),
        "RHS",
        fixed_line("", "RHS", "ROW A", "3."),
        "BOUNDS",
        fixed_line("UP", "BND", "COL 1", "10."),
        "ENDATA",
    ]) + "\n"
    p = tmp_path / "s.mps.gz"
    with gzip.open(p, "wt") as f:
        f.write(text)
    lp = read_mps(p)
    assert lp.col_names == ["COL 1"]
    assert lp.row_names == ["ROW A"]
    assert lp.rl[0] == 3.0 and lp.u[0] == 10.0


def test_unknown_row_is_an_error(tmp_path):
    p = tmp_path / "bad.mps"
    p.write_text("NAME X\nROWS\n N OBJ\nCOLUMNS\n    X1 NOPE 1.0\nENDATA\n")
    with pytest.raises(ValueError):
        read_mps(p)
