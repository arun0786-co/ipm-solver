"""Reader for MPS files (fixed and free format, optionally gzipped).

Supported sections: NAME, OBJSENSE, ROWS, COLUMNS, RHS, RANGES, BOUNDS,
ENDATA. Integer markers are ignored, so a MIP is read as its LP
relaxation. The objective constant follows the usual convention: a RHS
entry on the objective row gives -c0.
"""

import gzip

import numpy as np
import scipy.sparse as sp

from .model import LinearProgram


class MPSError(ValueError):
    pass


# 1-based column ranges of the six fields in fixed MPS
_FIXED = [(2, 3), (5, 12), (15, 22), (25, 36), (40, 47), (50, 61)]


def read_mps(path, fixed=None):
    """Read an MPS file into a LinearProgram.

    fixed=None tries free format first and falls back to fixed columns,
    which only matters for files whose names contain spaces.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="latin-1") as f:
        lines = f.read().splitlines()
    if fixed is None:
        try:
            return _parse(lines, fixed=False)
        except MPSError:
            return _parse(lines, fixed=True)
    return _parse(lines, fixed=fixed)


def _fields(line, fixed):
    if not fixed:
        return line.split()
    out = []
    for a, b in _FIXED:
        out.append(line[a - 1:b].strip())
    # the type field is blank in COLUMNS/RHS/RANGES lines
    return out


def _parse(lines, fixed):
    name = ""
    sense = 1.0
    obj_row = None
    row_type = {}          # row name -> 'E' | 'L' | 'G'
    row_index = {}
    row_names = []
    free_rows = set()      # extra N rows, dropped
    col_index = {}
    col_names = []
    obj = {}
    data_r, data_c, data_v = [], [], []
    rhs = {}
    ranges = {}
    bounds = []            # (type, col, value)
    section = None
    c0 = 0.0

    def add_col(cname):
        j = col_index.get(cname)
        if j is None:
            j = len(col_names)
            col_index[cname] = j
            col_names.append(cname)
        return j

    for raw in lines:
        if not raw.strip() or raw.startswith("*"):
            continue
        if not raw[0].isspace():
            head = raw.split()
            section = head[0].upper()
            if section == "NAME":
                name = head[1] if len(head) > 1 else ""
            elif section == "OBJSENSE" and len(head) > 1:
                sense = -1.0 if head[1].upper().startswith("MAX") else 1.0
            elif section == "ENDATA":
                break
            elif section not in ("ROWS", "COLUMNS", "RHS", "RANGES",
                                 "BOUNDS", "OBJSENSE", "OBJSENCE"):
                raise MPSError(f"unknown section {section!r}")
            continue

        if section in ("OBJSENSE", "OBJSENCE"):
            sense = -1.0 if raw.strip().upper().startswith("MAX") else 1.0
            continue

        if section == "ROWS":
            t = raw.split()
            if fixed:
                t = [raw[1:3].strip(), raw[4:12].strip()]
            if len(t) != 2:
                raise MPSError(f"bad ROWS line: {raw!r}")
            kind, rname = t[0].upper(), t[1]
            if kind == "N":
                if obj_row is None:
                    obj_row = rname
                else:
                    free_rows.add(rname)
            elif kind in ("E", "L", "G"):
                row_type[rname] = kind
                row_index[rname] = len(row_names)
                row_names.append(rname)
            else:
                raise MPSError(f"bad row type {kind!r}")
            continue

        f = _fields(raw, fixed)
        if fixed:
            f = f[1:]  # drop the type field for COLUMNS/RHS/RANGES
            if section == "BOUNDS":
                f = [raw[1:3].strip()] + f

        if section == "COLUMNS":
            if fixed:
                f = [x for x in f if x != ""]
            if "'MARKER'" in f:
                continue
            if len(f) not in (3, 5):
                raise MPSError(f"bad COLUMNS line: {raw!r}")
            j = add_col(f[0])
            for rname, val in zip(f[1::2], f[2::2]):
                v = float(val)
                if rname == obj_row:
                    obj[j] = obj.get(j, 0.0) + v
                elif rname in free_rows:
                    pass
                else:
                    i = row_index.get(rname)
                    if i is None:
                        raise MPSError(f"unknown row {rname!r}")
                    data_r.append(i)
                    data_c.append(j)
                    data_v.append(v)

        elif section in ("RHS", "RANGES"):
            if fixed:
                f = [x for x in f if x != ""]
            # the set name may be omitted in free MPS
            if len(f) in (2, 4):
                pairs = list(zip(f[0::2], f[1::2]))
            elif len(f) in (3, 5):
                pairs = list(zip(f[1::2], f[2::2]))
            else:
                raise MPSError(f"bad {section} line: {raw!r}")
            for rname, val in pairs:
                v = float(val)
                if section == "RHS":
                    if rname == obj_row:
                        c0 = -v
                    elif rname in free_rows:
                        pass
                    elif rname in row_index:
                        rhs[rname] = v
                    else:
                        raise MPSError(f"unknown row {rname!r}")
                else:
                    if rname not in row_index:
                        raise MPSError(f"unknown row {rname!r}")
                    ranges[rname] = v

        elif section == "BOUNDS":
            if fixed:
                kind = f[0].upper()
                cname = f[2]
                val = f[3]
                bounds.append((kind, cname, val))
                continue
            kind = f[0].upper()
            nval = kind not in ("FR", "MI", "PL", "BV")
            if nval and len(f) == 4:
                cname, val = f[2], f[3]
            elif nval and len(f) == 3:
                cname, val = f[1], f[2]
            elif not nval and len(f) == 3:
                cname, val = f[2], "0"
            elif not nval and len(f) == 2:
                cname, val = f[1], "0"
            elif not nval and len(f) == 4:
                cname, val = f[2], f[3]
            else:
                raise MPSError(f"bad BOUNDS line: {raw!r}")
            bounds.append((kind, cname, val))
        else:
            raise MPSError(f"data line outside a section: {raw!r}")

    m, n = len(row_names), len(col_names)
    A = sp.coo_matrix((data_v, (data_r, data_c)), shape=(m, n)).tocsr()
    A.sum_duplicates()
    c = np.zeros(n)
    for j, v in obj.items():
        c[j] = v

    rl = np.empty(m)
    ru = np.empty(m)
    for rname, i in row_index.items():
        b = rhs.get(rname, 0.0)
        t = row_type[rname]
        r = ranges.get(rname)
        if t == "E":
            if r is None or r == 0:
                rl[i] = ru[i] = b
            elif r > 0:
                rl[i], ru[i] = b, b + r
            else:
                rl[i], ru[i] = b + r, b
        elif t == "L":
            ru[i] = b
            rl[i] = -np.inf if r is None else b - abs(r)
        else:
            rl[i] = b
            ru[i] = np.inf if r is None else b + abs(r)

    l = np.zeros(n)
    u = np.full(n, np.inf)
    for kind, cname, val in bounds:
        j = col_index.get(cname)
        if j is None:
            raise MPSError(f"bound on unknown column {cname!r}")
        v = float(val) if val not in ("", None) else 0.0
        if kind == "UP":
            u[j] = v
            if v < 0 and l[j] == 0:
                l[j] = -np.inf
        elif kind == "LO":
            l[j] = v
        elif kind == "FX":
            l[j] = u[j] = v
        elif kind == "FR":
            l[j], u[j] = -np.inf, np.inf
        elif kind == "MI":
            l[j] = -np.inf
        elif kind == "PL":
            u[j] = np.inf
        elif kind == "BV":
            l[j], u[j] = 0.0, 1.0
        elif kind == "LI":
            l[j] = v
        elif kind == "UI":
            u[j] = v
        else:
            raise MPSError(f"unsupported bound type {kind!r}")

    return LinearProgram(c=sense * c, A=A, rl=rl, ru=ru, l=l, u=u,
                         c0=sense * c0, name=name,
                         row_names=row_names, col_names=col_names)
