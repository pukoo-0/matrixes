import json
import re
from collections import Counter
from fractions import Fraction

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

try:
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    HAS_PMG = True
except ImportError:
    HAS_PMG = False

# wide layout
st.set_page_config(page_title="Matrix transformation", page_icon="⚛️", layout="wide")

# css matching the point group app
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {background:#ffffff;}
.block-container {padding-top:2.4rem; padding-bottom:3rem; max-width:1320px;}
.stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .ui, .ui * {font-family:'Inter', system-ui, sans-serif;}
footer {visibility:hidden;}
header[data-testid="stHeader"] {background:transparent;}
[data-testid="stAppDeployButton"] {display:none;}
[data-testid="stFileUploaderDropzone"] {background:#f8fafc; border:1.5px dashed #d6dbe3; border-radius:12px;}
[data-testid="stExpander"] details {border:1px solid #eceef1; border-radius:12px;}
.stDownloadButton button {border-radius:10px; border:1px solid #e5e7eb; width:100%;}

.ui {color:#111827;}
.top {margin-bottom:20px;}
.title {font-size:30px; font-weight:700; letter-spacing:-.02em; margin:0;}
.subtitle {color:#6b7280; font-size:15px; margin:4px 0 0 0;}

.empty {text-align:center; padding:80px 20px; color:#6b7280; border:1px dashed #e2e5ea; border-radius:14px;}
.empty b {color:#111827; font-size:17px;}
</style>
""", unsafe_allow_html=True)

PLANE_ORDER = ["σh", "σv", "σv'", "σd", "σ"]
ATOMIC_MASS = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.18, "Na": 22.99, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Ti": 47.867, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "Ag": 107.87,
    "I": 126.90, "Pt": 195.08, "Au": 196.97,
}


def parse_xyz_fallback(xyz_str):
    lines = [l.strip() for l in xyz_str.strip().splitlines() if l.strip()]
    if not lines:
        raise ValueError("Empty XYZ file")
    try:
        n_atoms = int(lines[0].split()[0])
        atom_lines = lines[2:2 + n_atoms]
        if len(atom_lines) < n_atoms and len(lines) >= 1 + n_atoms:
            atom_lines = lines[1:1 + n_atoms]
    except ValueError:
        atom_lines = lines

    species, coords = [], []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        species.append(parts[0].capitalize())
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not species:
        raise ValueError("No valid atom coordinates found")

    coords = np.array(coords, dtype=float)
    masses = np.array([ATOMIC_MASS.get(el, 12.0) for el in species])
    com = np.average(coords, axis=0, weights=masses)
    centered = coords - com

    out_lines = [str(len(species)), "Centered"]
    for el, (x, y, z) in zip(species, centered):
        out_lines.append(f"{el:2s} {x:14.8f} {y:14.8f} {z:14.8f}")
    return species, centered, "\n".join(out_lines)


def is_sym_op(R, species, coords, tol):
    transformed = coords @ R.T
    for el in set(species):
        idx = [i for i, s in enumerate(species) if s == el]
        P = coords[idx]
        Q = transformed[idx]
        diff = Q[:, None, :] - P[None, :, :]
        dist = np.sqrt(np.min(np.sum(diff * diff, axis=-1), axis=1))
        if np.max(dist) > tol:
            return False
    return True


def fallback_symmetry(species, coords, tol):
    _, svals, _ = np.linalg.svd(coords)
    if len(coords) <= 2 or svals[1] < 1e-2:
        has_i = is_sym_op(-np.eye(3), species, coords, tol)
        return ("D*h" if has_i else "C*v"), []

    masses = np.array([ATOMIC_MASS.get(el, 12.0) for el in species])
    r2 = np.sum(coords ** 2, axis=1)
    I_tensor = np.sum(
        masses[:, None, None] * (r2[:, None, None] * np.eye(3) - coords[:, :, None] * coords[:, None, :]),
        axis=0
    )
    _, evecs = np.linalg.eigh(I_tensor)

    cand_dirs = [evecs[:, 0], evecs[:, 1], evecs[:, 2]]
    norms = np.linalg.norm(coords, axis=1)
    for p in coords[norms > 0.1][:36]:
        cand_dirs.append(p / np.linalg.norm(p))

    for el in set(species):
        idx = [i for i, s in enumerate(species) if s == el]
        sub_c = coords[idx]
        sub_r = np.round(np.linalg.norm(sub_c, axis=1), 1)
        for rad in np.unique(sub_r):
            if rad < 0.1:
                continue
            shell = sub_c[sub_r == rad][:24]
            for i in range(len(shell)):
                for j in range(i + 1, len(shell)):
                    for v in (shell[i] + shell[j], shell[i] - shell[j]):
                        if np.linalg.norm(v) > 0.1:
                            cand_dirs.append(v / np.linalg.norm(v))

    unique_dirs = []
    for u in cand_dirs:
        u = unit(u)
        if not any(abs(np.dot(u, v)) > 0.992 for v in unique_dirs):
            unique_dirs.append(u)

    valid_ops = [np.eye(3)]
    if is_sym_op(-np.eye(3), species, coords, tol):
        valid_ops.append(-np.eye(3))

    for u in unique_dirs:
        R_ref = np.eye(3) - 2.0 * np.outer(u, u)
        if is_sym_op(R_ref, species, coords, tol):
            if not any(np.allclose(R_ref, M, atol=5e-2) for M in valid_ops):
                valid_ops.append(R_ref)
        for n in (2, 3, 4, 5, 6, 8, 10, 12):
            for k in range(1, n):
                ang = 2.0 * np.pi * k / n
                R_rot = rodrigues(u, ang)
                if is_sym_op(R_rot, species, coords, tol):
                    if not any(np.allclose(R_rot, M, atol=5e-2) for M in valid_ops):
                        valid_ops.append(R_rot)
                R_imp = R_rot @ R_ref
                if is_sym_op(R_imp, species, coords, tol):
                    if not any(np.allclose(R_imp, M, atol=5e-2) for M in valid_ops):
                        valid_ops.append(R_imp)

    axes = get_axes(valid_ops, "C1", coords, tol)
    planes = get_planes(valid_ops)
    has_i = any(np.allclose(M, -np.eye(3), atol=5e-2) for M in valid_ops)
    if not axes:
        pg = "Cs" if planes else ("Ci" if has_i else "C1")
    else:
        nmax = max(a["n"] for a in axes)
        n_top = sum(1 for a in axes if a["n"] == nmax)
        if nmax == 5 and n_top >= 6:
            pg = "Ih" if planes else "I"
        elif nmax == 4 and n_top == 3:
            pg = "Oh" if planes else "O"
        elif nmax == 3 and sum(1 for a in axes if a["n"] == 3) == 4:
            pg = "Th" if has_i else ("Td" if planes else "T")
        else:
            main = [a["u"] for a in axes if a["n"] == nmax][0]
            perp_c2 = sum(1 for a in axes if a["n"] == 2 and abs(np.dot(a["u"], main)) < 0.15)
            has_sh = any(abs(np.dot(n, main)) > 0.98 for n in planes)
            if perp_c2 >= nmax:
                pg = f"D{nmax}h" if has_sh else (f"D{nmax}d" if planes else f"D{nmax}")
            else:
                pg = f"C{nmax}h" if has_sh else (f"C{nmax}v" if planes else f"C{nmax}")
    return pg, valid_ops


def sub(s):
    s = s.replace("*", "∞")
    if len(s) == 1:
        return s
    m = re.match(r"^([A-Za-zσ]+)(.+?)('+)?$", s)
    if m:
        base, sub_part, primes = m.group(1), m.group(2), m.group(3) or ""
        return f"{base}<sub>{sub_part}</sub>{primes}"
    return s[0] + "<sub>" + s[1:] + "</sub>"


def axis_name(n, subtype=None):
    if subtype:
        return subtype
    return "C∞" if n == 0 else f"C{n}"


def unit(x):
    norm = np.linalg.norm(x)
    return x / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


def clean_ops(ops, atol=5e-2):
    unique_matrices = []
    for op in ops:
        R = op.rotation_matrix
        U, _, Vt = np.linalg.svd(R)
        R_exact = U @ Vt
        if not any(np.allclose(R_exact, M, atol=atol) for M in unique_matrices):
            unique_matrices.append(R_exact)
    return unique_matrices


def rot_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def angle_to_order(a, max_denom=24):
    f = Fraction(float(a) / (2.0 * np.pi)).limit_denominator(max_denom)
    return f.denominator, f.numerator


def extract_axis(Q):
    a = rot_angle(Q)
    if a < 1e-3:
        return np.array([0.0, 0.0, 1.0])
    if abs(a - np.pi) > 1e-2:
        skew = np.array([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]])
        if np.linalg.norm(skew) > 1e-5:
            return unit(skew)
    M = Q + np.eye(3)
    norms = np.linalg.norm(M, axis=0)
    return unit(M[:, int(np.argmax(norms))])


def extract_improper_axis(R):
    return extract_axis(-R)


def rodrigues(u, a):
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def get_axes(matrices, pg, coords, tol):
    axes = []
    improper_orders = []
    for R in matrices:
        det = np.linalg.det(R)
        if det > 0:
            a = rot_angle(R)
            if a < 0.05:
                continue
            n, _ = angle_to_order(a)
            u = extract_axis(R)
            for ax in axes:
                if abs(np.dot(ax["u"], u)) > 0.99:
                    ax["n"] = max(ax["n"], n)
                    break
            else:
                axes.append({"u": u, "n": n, "sn": 0, "subtype": None})
        else:
            tr = np.trace(R)
            if abs(tr - 1.0) < 0.08 or abs(tr + 3.0) < 0.08:
                continue
            u = extract_improper_axis(R)
            Q = R @ (np.eye(3) - 2.0 * np.outer(u, u))
            a = rot_angle(Q)
            if a > 0.05:
                sn, _ = angle_to_order(a)
                improper_orders.append((u, sn))

    for u, sn in improper_orders:
        for ax in axes:
            if abs(np.dot(ax["u"], u)) > 0.99:
                ax["sn"] = max(ax["sn"], sn)
                break

    if not axes:
        return axes

    if pg.startswith("D") and pg not in ("D*h", "D2", "D2h"):
        nmax = max(a["n"] for a in axes)
        top = [a for a in axes if a["n"] == nmax]
        if len(top) > 1:
            top_s = [a for a in top if a["sn"] > nmax]
            main = top_s[0]["u"] if top_s else top[0]["u"]
        else:
            main = top[0]["u"]

        perp_c2 = [ax for ax in axes if ax["n"] == 2 and abs(np.dot(ax["u"], main)) < 0.1]
        if nmax % 2 == 0 and len(perp_c2) == nmax and nmax >= 4:
            on_ax = [
                int(np.sum(np.linalg.norm(coords - np.outer(coords @ ax["u"], ax["u"]), axis=1) < tol))
                for ax in perp_c2
            ]
            ref_idx = int(np.argmax(on_ax))
            ref_u = perp_c2[ref_idx]["u"]
            for ax in perp_c2:
                ang = np.arccos(np.clip(abs(np.dot(ax["u"], ref_u)), 0.0, 1.0))
                k = round(ang / (np.pi / nmax))
                ax["subtype"] = "C2'" if k % 2 == 0 else "C2''"
        elif len(perp_c2) == nmax:
            for ax in perp_c2:
                ax["subtype"] = "C2'"

    return axes


def get_planes(matrices):
    normals = []
    for R in matrices:
        if np.linalg.det(R) > 0 or abs(np.trace(R) - 1.0) > 0.08:
            continue
        n = extract_improper_axis(R)
        if all(abs(np.dot(n, m)) < 0.99 for m in normals):
            normals.append(n)
    return normals


def label_planes(normals, pg, axes, coords, tol):
    if not normals:
        return []
    if pg == "Cs":
        return ["σh"]
    if pg == "Ih" or not axes:
        return ["σ"] * len(normals)
    if pg == "Td":
        return ["σd"] * len(normals)
    if pg in ("Oh", "Th"):
        cube = [a["u"] for a in axes if a["n"] == (4 if pg == "Oh" else 2)]
        return ["σh" if any(abs(np.dot(n, u)) > 0.99 for u in cube) else "σd" for n in normals]

    nmax = max(a["n"] for a in axes)
    top = [a for a in axes if a["n"] == nmax]
    is_dnd = pg[0] == "D" and pg.endswith("d")
    if len(top) > 1 and not is_dnd:
        return ["σ"] * len(normals)
    if is_dnd:
        return ["σd"] * len(normals)
    main = top[0]["u"]

    labels = ["σh" if abs(np.dot(n, main)) > 0.99 else "σv" for n in normals]

    vert = [i for i, l in enumerate(labels) if l == "σv"]
    if nmax % 2 == 0 and len(vert) > 1:
        c2p = [a["u"] for a in axes if a.get("subtype") == "C2'"]
        on_plane = [int(np.sum(abs(coords @ normals[i]) < tol)) for i in vert]
        if max(on_plane) > min(on_plane):
            ref = normals[vert[int(np.argmax(on_plane))]]
        elif c2p:
            perp_scores = [max(abs(np.dot(normals[i], u)) for u in c2p) for i in vert]
            ref = normals[vert[int(np.argmin(perp_scores))]]
        else:
            ref = normals[vert[0]]

        other = "σv'" if nmax == 2 else "σd"
        for i in vert:
            ang = np.arccos(np.clip(abs(np.dot(normals[i], ref)), 0.0, 1.0))
            k = round(ang / (np.pi / nmax))
            labels[i] = "σv" if k % 2 == 0 else other
    return labels


def build_linear_sym_ops(pg, coords):
    main = unit(np.linalg.svd(coords - coords.mean(0))[2][0])
    a = np.array([1.0, 0.0, 0.0]) if abs(main[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    p1 = unit(a - np.dot(a, main) * main)
    p2 = np.cross(main, p1)

    def clean(M):
        return np.where(np.abs(M) < 1e-6, 0.0, np.round(M, 4)).tolist()

    ops = [
        {"kind": "C", "u": main.tolist(), "a": float(np.pi / 2), "name": "C<sub>∞</sub> (90°)",
         "group": "Rotations", "R": clean(rodrigues(main, np.pi / 2))},
        {"kind": "C", "u": main.tolist(), "a": float(np.pi), "name": "C<sub>∞</sub> (180°)",
         "group": "Rotations", "R": clean(rodrigues(main, np.pi))},
        {"kind": "s", "n": p1.tolist(), "name": "σ<sub>v</sub> 1",
         "group": "Reflections", "R": clean(np.eye(3) - 2.0 * np.outer(p1, p1))},
        {"kind": "s", "n": p2.tolist(), "name": "σ<sub>v</sub> 2",
         "group": "Reflections", "R": clean(np.eye(3) - 2.0 * np.outer(p2, p2))},
    ]
    if pg == "D*h":
        ops.extend([
            {"kind": "C", "u": p1.tolist(), "a": float(np.pi), "name": "C<sub>2</sub>'",
             "group": "Rotations", "R": clean(rodrigues(p1, np.pi))},
            {"kind": "i", "name": "i", "group": "Inversion", "R": clean(-np.eye(3))},
            {"kind": "s", "n": main.tolist(), "name": "σ<sub>h</sub>",
             "group": "Reflections", "R": clean(np.eye(3) - 2.0 * np.outer(main, main))},
        ])
    return ops


def build_sym_ops(matrices, axes, planes):
    out = []
    for R in matrices:
        det, tr = np.linalg.det(R), np.trace(R)
        R_clean = np.where(np.abs(R) < 1e-6, 0.0, np.round(R, 4)).tolist()

        if det < 0 and abs(tr - 1.0) < 0.08:
            if not planes:
                continue
            n = extract_improper_axis(R)
            k = int(np.argmax([abs(np.dot(p["n"], n)) for p in planes]))
            p = planes[k]
            out.append({
                "kind": "s", "n": p["n"].tolist(), "name": p["name"], "group": "Reflections",
                "R": R_clean, "sort": (3, k, 0, 0, 0)
            })
            continue

        if det < 0 and abs(tr + 3.0) < 0.08:
            out.append({
                "kind": "i", "name": "i", "group": "Inversion",
                "R": R_clean, "sort": (1, 0, 0, 0, 0)
            })
            continue

        if det > 0:
            u = extract_axis(R)
            Q = R
        else:
            u = extract_improper_axis(R)
            Q = R @ (np.eye(3) - 2.0 * np.outer(u, u))

        a = rot_angle(Q)
        if a < 0.05:
            continue
        if not np.allclose(rodrigues(u, a), Q, atol=0.05):
            a = -a

        j = int(np.argmax([abs(np.dot(ax["u"], u)) for ax in axes])) if axes else -1
        if j >= 0 and np.dot(axes[j]["u"], u) < 0:
            u, a = -u, -a
        a = float(a % (2.0 * np.pi))

        n, k = angle_to_order(a)
        letter = "C" if det > 0 else "S"
        primes = ""
        if letter == "C" and n == 2 and j >= 0 and axes[j].get("subtype"):
            primes = axes[j]["subtype"][2:]

        if k == 1:
            name = f"{letter}<sub>{n}</sub>{primes}"
        else:
            name = f"{letter}<span class='ss'><sup>{k}</sup><sub>{n}</sub></span>{primes}"

        if j >= 0:
            nax = axes[j]["n"]
            sub_ax = axes[j].get("subtype")
            same = [i for i, ax in enumerate(axes) if ax["n"] == nax and ax.get("subtype") == sub_ax]
            num = f" {same.index(j) + 1}" if len(same) > 1 else ""
            ax_label = sub(axis_name(nax, sub_ax))
            if nax != n or letter == "S":
                name += f" (on {ax_label} axis{num})"
            elif num:
                name += f" (axis{num})"

        out.append({
            "kind": letter, "u": u.tolist(), "a": a, "name": name, "R": R_clean,
            "group": "Rotations" if letter == "C" else "Improper rotations",
            "sort": (0 if letter == "C" else 2, -n, len(primes), int(j >= 0 and axes[j]["n"] != n), j + k * 0.01)
        })

    out.sort(key=lambda o: o["sort"])
    for o in out:
        del o["sort"]
    return out


def atom_labels_from_species(species):
    count, labs = Counter(), []
    for el in species:
        count[el] += 1
        labs.append(f"{el}{count[el]}")
    return labs


def render_app(xyz, coords, species, pg, sym_ops, order_str):
    size = float(max(np.linalg.norm(coords, axis=1).max(), 1.2))
    atoms = [
        {"p": c.tolist(), "el": el, "lab": lab}
        for c, el, lab in zip(coords, species, atom_labels_from_species(species))
    ]

    data = {
        "xyz": xyz,
        "pg": sub(pg),
        "orderStr": order_str,
        "atoms": atoms,
        "symOps": sym_ops,
        "R": size * 1.08,
    }

    html = r"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
* {box-sizing:border-box; font-family:'Inter', system-ui, sans-serif;}
body {margin:0; background:#fff; color:#111827;}

.layout {display:grid; grid-template-columns:1fr 380px; gap:20px; align-items:start;}
@media (max-width: 1024px) { .layout {grid-template-columns:1fr;} }

.card {background:#fff; border:1px solid #eceef1; border-radius:14px; overflow:hidden;
       box-shadow:0 1px 2px rgba(16,24,40,.04); margin-bottom:12px;}
.pad {padding:18px 20px;}

.cap {font-size:13px; font-weight:500; color:#6b7280; margin:0;}
.pg-row {display:flex; align-items:baseline; justify-content:space-between;}
.pg-symbol {font-family:Georgia, 'Times New Roman', serif; font-size:48px; line-height:1.05; margin:4px 0 0 0;}
.pg-order {font-size:13px; color:#6b7280;}

.bar {display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:space-between;
      padding:10px 12px; border-bottom:1px solid #f1f2f4; background:#fff;}
.seg {display:inline-flex; background:#f3f4f6; border-radius:9px; padding:3px;}
.seg button {border:0; background:transparent; padding:5px 11px; font-size:12.5px; border-radius:7px;
             cursor:pointer; color:#4b5563;}
.seg button.on {background:#fff; color:#111827; box-shadow:0 1px 2px rgba(0,0,0,.1); font-weight:500;}

.group {display:inline-flex; gap:6px; flex-wrap:wrap; align-items:center;}
.chip {border:1px solid #e5e7eb; background:#fff; border-radius:999px; padding:4px 11px; font-size:12.5px;
       cursor:pointer; color:#374151; display:inline-flex; align-items:center; gap:6px;}
.chip:hover {border-color:#cbd0d8;}
.chip.off {color:#a1a7b0; background:#fafafa;}
.chip .d {width:8px; height:8px; border-radius:50%;}
.chip.off .d {opacity:.25;}

#vwrap {position:relative; width:100%; height:540px; background:#0b101b; overflow:hidden;}
#vwrap.light {background:#ffffff;}
#v {width:100%; height:100%; position:relative;}
#gridCv {position:absolute; inset:0; width:100%; height:100%; pointer-events:none; z-index:4;}
#ovLabs {position:absolute; inset:0; pointer-events:none; overflow:hidden; z-index:5;}

.btag {position:absolute; left:0; top:0; font-size:12px; font-weight:600; color:#fff; padding:1px 6px;
       border-radius:6px; white-space:nowrap; will-change:transform;}

.hud {position:absolute; left:12px; top:12px; z-index:6; padding:6px 10px; border-radius:8px; font-size:12px;
      background:rgba(15, 23, 42, 0.72); color:#e5e7eb; font-variant-numeric:tabular-nums;
      border:1px solid rgba(255,255,255,.1); pointer-events:none;}
#vwrap.light .hud {background:rgba(255,255,255,.88); color:#374151; border-color:#e5e7eb;}

.anim {padding:12px; border-top:1px solid #f1f2f4; background:#fafbfc;}
.anim-row {display:flex; flex-wrap:wrap; gap:8px; align-items:center;}
.play {border:0; background:#111827; color:#fff; border-radius:9px; padding:6px 14px; font-size:13px;
       font-weight:500; cursor:pointer; min-width:72px;}
.anim input[type=range] {flex:1; min-width:130px; accent-color:#111827; cursor:pointer;}

/* Dropdown */
.dd {position:relative; width:100%; margin:8px 0 14px 0;}
.dd-btn {display:flex; width:100%; align-items:center; justify-content:space-between; gap:10px; font-size:13px;
         padding:7px 11px; border:1px solid #e5e7eb; border-radius:9px; background:#fff; color:#111827;
         cursor:pointer;}
.dd-btn:hover {border-color:#cbd0d8;}
.dd-menu {display:none; position:absolute; left:0; top:calc(100% + 4px); z-index:30; width:100%; max-height:260px;
          overflow:auto; background:#fff; border:1px solid #e5e7eb; border-radius:10px;
          box-shadow:0 8px 24px rgba(16,24,40,.12); padding:4px;}
.dd.open .dd-menu {display:block;}
.dd-group {font-size:11.5px; color:#6b7280; font-weight:600; padding:8px 10px 4px;}
.dd-item {display:block; width:100%; text-align:left; border:0; background:none; padding:6px 10px; font-size:13px;
          border-radius:7px; cursor:pointer; color:#111827;}
.dd-item:hover {background:#f3f4f6;}
.dd-item.sel {background:#eef2ff; font-weight:500;}
sub, sup {font-size:.72em; line-height:0;}
.ss {display:inline-flex; flex-direction:column; vertical-align:middle; font-size:.7em; line-height:1; margin-left:1px;}
.ss sup, .ss sub {font-size:1em; line-height:1; vertical-align:baseline;}

/* Matrix Bracket */
.mat-box {display:flex; align-items:center; justify-content:center; gap:6px; margin:6px 0 12px 0;}
.bkt {width:7px; align-self:stretch; border:1.8px solid #111827;}
.bkt.l {border-right:0; border-radius:3px 0 0 3px;}
.bkt.r {border-left:0; border-radius:0 3px 3px 0;}
.mgrid {display:grid; grid-template-columns:repeat(3, 72px); gap:6px 8px; padding:3px;}
.mhdr {display:grid; grid-template-columns:repeat(3, 72px); gap:0 8px; text-align:center; font-size:12px; font-weight:600; margin-bottom:2px;}

.minp {width:100%; padding:5px 4px; font-size:13.5px; font-weight:500; text-align:center; border-radius:8px;
       border:1px solid #e5e7eb; font-variant-numeric:tabular-nums; outline:none; background:#fff; color:#111827;}
.minp:focus {border-color:#2563eb;}
.minp.ci {border-bottom:2px solid #16a34a;}
.minp.cj {border-bottom:2px solid #dc2626;}
.minp.ck {border-bottom:2px solid #2563eb;}

.ci-txt {color:#16a34a;}
.cj-txt {color:#dc2626;}
.ck-txt {color:#2563eb;}
.cv-txt {color:#d97706;}

.presets {display:flex; flex-wrap:wrap; gap:6px; margin-top:8px;}
.pbtn {border:1px solid #e5e7eb; background:#fff; border-radius:8px; padding:4px 10px; font-size:12px;
       color:#374151; cursor:pointer;}
.pbtn:hover {border-color:#cbd0d8; background:#f9fafb;}

.stats {display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin-top:10px;}
.stat {border:1px solid #eceef1; border-radius:10px; padding:10px 12px; background:#fafbfc;}
.stat .v {font-size:17px; font-weight:600; margin-top:2px; font-variant-numeric:tabular-nums;}

.vec-row {display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;}
select.small {font-size:12.5px; padding:4px 8px; border:1px solid #e5e7eb; border-radius:8px; background:#fff; color:#111827;}
.vec-out {font-size:13px; color:#374151; font-variant-numeric:tabular-nums; line-height:1.6;
          padding:8px 10px; background:#fafbfc; border:1px solid #eceef1; border-radius:10px;}
</style>

<div class="layout">
  <!-- LEFT: 3D Viewer -->
  <div>
    <div class="card" style="margin-bottom:0;">
      <div class="bar">
        <div class="seg" id="styleSeg"></div>
        <div class="group">
          <button class="chip" id="tGrid"><span class="d" style="background:#38bdf8"></span>Grid</button>
          <button class="chip" id="tBgGrid"><span class="d" style="background:#94a3b8"></span>Base grid</button>
          <button class="chip" id="tCube"><span class="d" style="background:#22c55e"></span>Unit cube</button>
          <button class="chip" id="tGhost"><span class="d" style="background:#9ca3af"></span>Start positions</button>
          <button class="chip" id="tDark">Dark</button>
          <button class="chip" id="tReset">Reset view</button>
        </div>
      </div>

      <div id="vwrap">
        <div id="v"></div>
        <canvas id="gridCv"></canvas>
        <div id="ovLabs"></div>
        <div class="hud" id="hudBox"></div>
      </div>

      <div class="anim">
        <div class="anim-row">
          <button class="play" id="playBtn">Pause</button>
          <input type="range" id="scrub" min="0" max="1000" value="0" aria-label="Progress">
          <div class="seg">
            <button id="mLin" class="on">Linear</button>
            <button id="mGeod">Rigid</button>
          </div>
          <button class="chip" id="tLoop">Loop</button>
        </div>
      </div>
    </div>
  </div>

  <!-- RIGHT: Controls -->
  <div>
    <div class="card pad">
      <div class="pg-row">
        <div>
          <p class="cap">Point group</p>
          <p class="pg-symbol" id="pgSym"></p>
        </div>
        <span class="pg-order" id="pgOrd"></span>
      </div>
    </div>

    <div class="card pad">
      <p class="cap">Symmetry operation</p>
      <div class="dd" id="symDD"></div>

      <p class="cap" style="margin-bottom:6px;">Transformation matrix M</p>
      <div style="display:flex; flex-direction:column; align-items:center;">
        <div class="mhdr">
          <span class="ci-txt">î</span>
          <span class="cj-txt">ĵ</span>
          <span class="ck-txt">k̂</span>
        </div>
        <div class="mat-box">
          <div class="bkt l"></div>
          <div class="mgrid" id="matInputs"></div>
          <div class="bkt r"></div>
        </div>
      </div>

      <p class="cap">Presets</p>
      <div class="presets" id="presetBox"></div>

      <div class="stats">
        <div class="stat"><p class="cap">det(M)</p><div class="v" id="detTarget">1.00</div></div>
        <div class="stat"><p class="cap">det M(t)</p><div class="v" id="detLive">1.00</div></div>
        <div class="stat"><p class="cap">RMSD</p><div class="v" id="rmsdVal">0.00 Å</div></div>
      </div>
    </div>

    <div class="card pad">
      <div class="vec-row">
        <p class="cap">Atom vector</p>
        <select id="atomSel" class="small"></select>
      </div>
      <div class="vec-out" id="atomEqBox"></div>
    </div>
  </div>
</div>

<script>
const D = __DATA__;
const $ = id => document.getElementById(id);

$("pgSym").innerHTML = D.pg;
$("pgOrd").textContent = `h = ${D.orderStr}`;

const STYLES = {
  "Ball & stick": {stick: {radius: 0.14}, sphere: {scale: 0.25}},
  "Sticks": {stick: {radius: 0.18}},
  "Spacefill": {sphere: {scale: 0.9}}
};

const PRESETS = [
  {name: "Identity", M: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
  {name: "Shear XY", M: [[1, 0.85, 0], [0.45, 1, 0], [0, 0, 1]]},
  {name: "Scale Z", M: [[1, 0, 0], [0, 1, 0], [0, 0, 1.6]]},
  {name: "Project XY (det = 0)", M: [[1, 0, 0], [0, 1, 0], [0, 0, 0]]},
  {name: "Rank 1 (det = 0)", M: [[1, 0.5, 0.5], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]]}
];

const hasSymOps = D.symOps.length > 0;
const S = {
  style: "Ball & stick",
  dark: true,
  showGrid: true,
  showBgGrid: true,
  showCube: true,
  showGhost: true,
  linearInterp: true,
  symIdx: hasSymOps ? 0 : -1,
  M: hasSymOps ? D.symOps[0].R.map(r => [...r]) : PRESETS[1].M.map(r => [...r]),
  atomIdx: 0,
  t: 0,
  playing: true,
  loop: true
};

const v = $3Dmol.createViewer("v", {backgroundColor: "#0b101b"});
const model = v.addModel(D.xyz, "xyz");
const atoms = model.selectedAtoms({});
const P0 = D.atoms.map(a => a.p);
const ghost = v.addModel(D.xyz, "xyz");

function updateGhostStyle() {
  const g = {color: S.dark ? "#64748b" : "#9ca3af", opacity: 0.28};
  if (!S.showGhost) { ghost.setStyle({}, {}); return; }
  if (S.style === "Spacefill") ghost.setStyle({}, {sphere: {...g, scale: 0.92}});
  else if (S.style === "Sticks") ghost.setStyle({}, {sphere: {...g, radius: 0.24}});
  else ghost.setStyle({}, {sphere: {...g, scale: 0.3}});
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const add = (a, b, s = 1) => [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const ease = x => x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
const lerp = (a, b, s) => a + (b - a) * s;

function rotate(p, u, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return add(add(mul(p, c), cross(u, p), s), u, dot(u, p) * (1 - c));
}
function reflect(p, n, s) { return add(p, n, -2 * s * dot(p, n)); }

function getLiveMatrix() {
  const s = ease(S.t);
  const op = S.symIdx >= 0 ? D.symOps[S.symIdx] : null;

  if (!S.linearInterp && op) {
    const transformPt = p => {
      if (op.kind === "C") return rotate(p, op.u, op.a * s);
      if (op.kind === "s") return reflect(p, op.n, s);
      if (op.kind === "i") return mul(p, 1 - 2 * s);
      if (S.t < 0.5) return rotate(p, op.u, op.a * ease(2 * S.t));
      return reflect(rotate(p, op.u, op.a), op.u, ease(2 * S.t - 1));
    };
    const c1 = transformPt([1, 0, 0]);
    const c2 = transformPt([0, 1, 0]);
    const c3 = transformPt([0, 0, 1]);
    return [
      [c1[0], c2[0], c3[0]],
      [c1[1], c2[1], c3[1]],
      [c1[2], c2[2], c3[2]]
    ];
  }

  return [
    [lerp(1, S.M[0][0], s), lerp(0, S.M[0][1], s), lerp(0, S.M[0][2], s)],
    [lerp(0, S.M[1][0], s), lerp(1, S.M[1][1], s), lerp(0, S.M[1][2], s)],
    [lerp(0, S.M[2][0], s), lerp(0, S.M[2][1], s), lerp(1, S.M[2][2], s)]
  ];
}

function applyMat(M, p) {
  return [
    M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2],
    M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2],
    M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2]
  ];
}

function det3(M) {
  return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
       - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
       + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
}

function computeSymmetryRMSD(M) {
  let sumSq = 0;
  for (let i = 0; i < P0.length; i++) {
    const q = applyMat(M, P0[i]);
    const el = D.atoms[i].el;
    let bestD2 = 1e9;
    for (let j = 0; j < P0.length; j++) {
      if (D.atoms[j].el !== el) continue;
      const dx = q[0] - P0[j][0], dy = q[1] - P0[j][1], dz = q[2] - P0[j][2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bestD2) bestD2 = d2;
    }
    sumSq += bestD2;
  }
  return Math.sqrt(sumSq / P0.length);
}

const GRID_SEGS = [];
const CUBE_CORNERS = [
  [0,0,0], [1,0,0], [0,1,0], [0,0,1],
  [1,1,0], [1,0,1], [0,1,1], [1,1,1]
];
const CUBE_EDGES = [
  [0,1],[0,2],[0,3], [1,4],[1,5], [2,4],[2,6], [3,5],[3,6], [4,7],[5,7],[6,7]
];
const CUBE_FACES = [
  [0,1,4,2], [3,5,7,6], [0,1,5,3], [2,4,7,6], [0,2,6,3], [1,4,7,5]
];
(function initLattice() {
  const ticks = [-1, -0.5, 0, 0.5, 1];
  for (const a of ticks) {
    GRID_SEGS.push([[-1, a, 0], [1, a, 0]], [[a, -1, 0], [a, 1, 0]]);
    GRID_SEGS.push([[-1, 0, a], [1, 0, a]], [[a, 0, -1], [a, 0, 1]]);
    GRID_SEGS.push([[0, -1, a], [0, 1, a]], [[0, a, -1], [0, a, 1]]);
  }
  for (const s of [-1, 1]) {
    for (const t of [-1, 1]) {
      GRID_SEGS.push([[-1, s, t], [1, s, t]], [[s, -1, t], [s, 1, t]], [[s, t, -1], [s, t, 1]]);
    }
  }
})();

const ovLabs = $("ovLabs");
const bTags = [];
[["î", "#16a34a"], ["ĵ", "#dc2626"], ["k̂", "#2563eb"]].forEach(([txt, col]) => {
  const s = document.createElement("span");
  s.className = "btag";
  s.style.background = col;
  s.textContent = txt;
  ovLabs.appendChild(s);
  bTags.push(s);
});

const gridCv = $("gridCv");
const ctx = gridCv.getContext("2d");

function drawArrow2D(x0, y0, x1, y1, color, width, dashed = false) {
  const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy);
  if (len < 2) return;
  const ux = dx / len, uy = dy / len;
  const headL = Math.min(12, Math.max(7, len * 0.3)), headW = headL * 0.56;
  ctx.save();
  if (dashed) ctx.setLineDash([4, 3]);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1 - ux * headL * 0.65, y1 - uy * headL * 0.65);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - ux * headL - uy * headW, y1 - uy * headL + ux * headW);
  ctx.lineTo(x1 - ux * headL + uy * headW, y1 - uy * headL - ux * headW);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function renderOverlay() {
  const cv3d = $("v").querySelector("canvas");
  if (!cv3d) return;
  const r = cv3d.getBoundingClientRect();
  const left = r.left + window.pageXOffset - document.documentElement.clientLeft;
  const top = r.top + window.pageYOffset - document.documentElement.clientTop;
  const width = r.width, height = r.height;

  const dpr = window.devicePixelRatio || 1;
  if (gridCv.width !== Math.round(width * dpr) || gridCv.height !== Math.round(height * dpr)) {
    gridCv.width = Math.round(width * dpr);
    gridCv.height = Math.round(height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const Mt = getLiveMatrix();
  const ih = [Mt[0][0], Mt[1][0], Mt[2][0]];
  const jh = [Mt[0][1], Mt[1][1], Mt[2][1]];
  const kh = [Mt[0][2], Mt[1][2], Mt[2][2]];
  const E1 = [1, 0, 0], E2 = [0, 1, 0], E3 = [0, 0, 1];
  const L = D.R;

  const linComb = (c, b1, b2, b3, scale = L) => ({
    x: scale * (c[0] * b1[0] + c[1] * b2[0] + c[2] * b3[0]),
    y: scale * (c[0] * b1[1] + c[1] * b2[1] + c[2] * b3[1]),
    z: scale * (c[0] * b1[2] + c[1] * b2[2] + c[2] * b3[2])
  });

  const pts3D = [];
  pts3D.push({x: 0, y: 0, z: 0});
  pts3D.push(linComb(E1, E1, E2, E3), linComb(E2, E1, E2, E3), linComb(E3, E1, E2, E3));
  pts3D.push(linComb(E1, ih, jh, kh), linComb(E2, ih, jh, kh), linComb(E3, ih, jh, kh));
  pts3D.push(linComb([1.12, 0, 0], ih, jh, kh), linComb([0, 1.12, 0], ih, jh, kh), linComb([0, 0, 1.12], ih, jh, kh));

  const bgStart = pts3D.length;
  GRID_SEGS.forEach(([a, b]) => {
    pts3D.push(linComb(a, E1, E2, E3), linComb(b, E1, E2, E3));
  });

  const fgStart = pts3D.length;
  GRID_SEGS.forEach(([a, b]) => {
    pts3D.push(linComb(a, ih, jh, kh), linComb(b, ih, jh, kh));
  });

  const cubeStart = pts3D.length;
  CUBE_CORNERS.forEach(c => pts3D.push(linComb(c, ih, jh, kh, L * 0.55)));

  const trackStart = pts3D.length;
  if (S.atomIdx >= 0 && S.atomIdx < P0.length) {
    const [x, y, z] = P0[S.atomIdx];
    pts3D.push(
      linComb([x, 0, 0], ih, jh, kh, 1.0),
      linComb([x, y, 0], ih, jh, kh, 1.0),
      linComb([x, y, z], ih, jh, kh, 1.0)
    );
  }

  const sc = v.modelToScreen(pts3D).map(p => ({x: p.x - left, y: p.y - top}));
  const O = sc[0];

  if (S.showBgGrid) {
    ctx.lineWidth = 0.85;
    ctx.strokeStyle = S.dark ? "rgba(148, 163, 184, 0.14)" : "rgba(100, 116, 139, 0.16)";
    ctx.beginPath();
    for (let i = 0; i < GRID_SEGS.length; i++) {
      const p1 = sc[bgStart + 2 * i], p2 = sc[bgStart + 2 * i + 1];
      ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
    }
    ctx.stroke();
  }

  if (S.showGrid) {
    ctx.lineWidth = 1.15;
    ctx.strokeStyle = S.dark ? "rgba(56, 189, 248, 0.36)" : "rgba(2, 132, 199, 0.32)";
    ctx.beginPath();
    for (let i = 0; i < GRID_SEGS.length; i++) {
      const p1 = sc[fgStart + 2 * i], p2 = sc[fgStart + 2 * i + 1];
      ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
    }
    ctx.stroke();
  }

  if (S.showCube) {
    const d = det3(Mt);
    ctx.fillStyle = Math.abs(d) < 0.05
      ? "rgba(245, 158, 11, 0.20)"
      : (d > 0 ? "rgba(34, 197, 94, 0.13)" : "rgba(239, 68, 68, 0.15)");
    CUBE_FACES.forEach(([a, b, c, dIdx]) => {
      const pa = sc[cubeStart + a], pb = sc[cubeStart + b],
            pc = sc[cubeStart + c], pd = sc[cubeStart + dIdx];
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y);
      ctx.lineTo(pc.x, pc.y); ctx.lineTo(pd.x, pd.y);
      ctx.closePath();
      ctx.fill();
    });
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = Math.abs(d) < 0.05 ? "#f59e0b" : (d > 0 ? "#22c55e" : "#ef4444");
    ctx.beginPath();
    CUBE_EDGES.forEach(([a, b]) => {
      const p1 = sc[cubeStart + a], p2 = sc[cubeStart + b];
      ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
    });
    ctx.stroke();
  }

  if (S.showGrid) {
    drawArrow2D(O.x, O.y, sc[4].x, sc[4].y, "#22c55e", 3.2);
    drawArrow2D(O.x, O.y, sc[5].x, sc[5].y, "#ef4444", 3.2);
    drawArrow2D(O.x, O.y, sc[6].x, sc[6].y, "#3b82f6", 3.2);
    [sc[7], sc[8], sc[9]].forEach((p, idx) => {
      bTags[idx].style.display = "block";
      bTags[idx].style.transform = `translate(${p.x}px,${p.y}px) translate(-50%,-50%)`;
    });
  } else {
    bTags.forEach(t => { t.style.display = "none"; });
  }

  if (S.atomIdx >= 0 && S.atomIdx < P0.length) {
    const pX = sc[trackStart], pXY = sc[trackStart + 1], pXYZ = sc[trackStart + 2];
    drawArrow2D(O.x, O.y, pX.x, pX.y, "rgba(34, 197, 94, 0.85)", 2.1, true);
    drawArrow2D(pX.x, pX.y, pXY.x, pXY.y, "rgba(239, 68, 68, 0.85)", 2.1, true);
    drawArrow2D(pXY.x, pXY.y, pXYZ.x, pXYZ.y, "rgba(59, 130, 246, 0.85)", 2.1, true);
    drawArrow2D(O.x, O.y, pXYZ.x, pXYZ.y, "#f59e0b", 3.4, false);
  }
}

v.setViewChangeCallback(renderOverlay);

function updateAtoms() {
  const Mt = getLiveMatrix();
  atoms.forEach((a, i) => {
    const q = applyMat(Mt, P0[i]);
    a.x = q[0]; a.y = q[1]; a.z = q[2];
  });
  if (!(model.syncAtomPositions && model.syncAtomPositions())) {
    model.setStyle({}, STYLES[S.style]);
  }
  v.render();
}

function fullRedraw() {
  v.setBackgroundColor(S.dark ? "#0b101b" : "#ffffff");
  $("vwrap").classList.toggle("light", !S.dark);
  updateGhostStyle();
  model.setStyle({}, STYLES[S.style]);
  updateAtoms();
  updateUI();
}

function buildSymDropdown() {
  const root = $("symDD");
  root.innerHTML = '<button type="button" class="dd-btn"><span class="dd-val"></span><span style="color:#9ca3af;font-size:11px">▾</span></button><div class="dd-menu"></div>';
  const btn = root.querySelector(".dd-btn"), menu = root.querySelector(".dd-menu"), valSpan = root.querySelector(".dd-val");

  if (!hasSymOps) {
    const h = document.createElement("div");
    h.className = "dd-group";
    h.textContent = "Only E";
    menu.appendChild(h);
  } else {
    const groups = {};
    D.symOps.forEach((o, i) => {
      if (!groups[o.group]) groups[o.group] = [];
      groups[o.group].push({idx: i, html: o.name});
    });

    Object.entries(groups).forEach(([gTitle, items]) => {
      const h = document.createElement("div");
      h.className = "dd-group"; h.textContent = gTitle;
      menu.appendChild(h);
      items.forEach(it => {
        const b = document.createElement("button");
        b.type = "button"; b.className = "dd-item"; b.innerHTML = it.html;
        b.dataset.idx = String(it.idx);
        b.onclick = () => {
          root.classList.remove("open");
          S.symIdx = it.idx;
          S.M = D.symOps[it.idx].R.map(r => [...r]);
          syncMatrixInputs();
          S.t = 0; S.playing = true;
          startAnim();
          updateUI();
        };
        menu.appendChild(b);
      });
    });
  }

  btn.onclick = e => { e.stopPropagation(); root.classList.toggle("open"); };
  document.addEventListener("click", () => root.classList.remove("open"));

  return {
    refresh() {
      if (S.symIdx >= 0 && D.symOps[S.symIdx]) {
        valSpan.innerHTML = D.symOps[S.symIdx].name;
      } else {
        valSpan.innerHTML = "Custom matrix";
      }
      menu.querySelectorAll(".dd-item").forEach(b => {
        b.classList.toggle("sel", Number(b.dataset.idx) === S.symIdx);
      });
    }
  };
}
const symDD = buildSymDropdown();

function buildMatrixEditor() {
  const grid = $("matInputs");
  const colCls = ["ci", "cj", "ck"];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const inp = document.createElement("input");
      inp.type = "number"; inp.step = "0.1";
      inp.className = "minp " + colCls[c];
      inp.id = `m_${r}_${c}`;
      inp.oninput = () => {
        const val = parseFloat(inp.value);
        if (!isNaN(val)) {
          S.M[r][c] = val;
          S.symIdx = -1;
          updateAtoms();
          updateUI();
        }
      };
      grid.appendChild(inp);
    }
  }

  const pBox = $("presetBox");
  PRESETS.forEach(p => {
    const b = document.createElement("button");
    b.className = "pbtn";
    b.textContent = p.name;
    b.onclick = () => {
      S.M = p.M.map(r => [...r]);
      S.symIdx = -1;
      syncMatrixInputs();
      S.t = 0; S.playing = true;
      startAnim();
      updateUI();
    };
    pBox.appendChild(b);
  });

  const aSel = $("atomSel");
  aSel.innerHTML = '<option value="-1">Off</option>';
  D.atoms.forEach((a, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = a.lab;
    aSel.appendChild(opt);
  });
  aSel.value = "0";
  aSel.onchange = () => { S.atomIdx = Number(aSel.value); updateAtoms(); updateUI(); };
}

function syncMatrixInputs() {
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const el = $(`m_${r}_${c}`);
      if (el && document.activeElement !== el) el.value = Number(S.M[r][c].toFixed(3));
    }
  }
}

function updateUI() {
  [...$("styleSeg").children].forEach(b => b.classList.toggle("on", b.textContent === S.style));
  $("tGrid").classList.toggle("off", !S.showGrid);
  $("tBgGrid").classList.toggle("off", !S.showBgGrid);
  $("tCube").classList.toggle("off", !S.showCube);
  $("tGhost").classList.toggle("off", !S.showGhost);
  $("tDark").textContent = S.dark ? "Dark" : "Light";
  $("mLin").classList.toggle("on", S.linearInterp);
  $("mGeod").classList.toggle("on", !S.linearInterp);
  $("tLoop").classList.toggle("off", !S.loop);
  $("playBtn").textContent = S.playing ? "Pause" : "Play";
  $("scrub").value = Math.round(S.t * 1000);

  symDD.refresh();

  const Mt = getLiveMatrix();
  const dtTarget = det3(S.M);
  const dtLive = det3(Mt);
  const rmsd = computeSymmetryRMSD(S.M);

  $("detTarget").textContent = dtTarget.toFixed(2);
  $("detLive").textContent = dtLive.toFixed(2);
  $("rmsdVal").textContent = `${rmsd.toFixed(2)} Å`;

  $("hudBox").innerHTML =
    `<span style="color:#4ade80">î [${Mt[0][0].toFixed(2)}, ${Mt[1][0].toFixed(2)}, ${Mt[2][0].toFixed(2)}]</span> &nbsp; ` +
    `<span style="color:#f87171">ĵ [${Mt[0][1].toFixed(2)}, ${Mt[1][1].toFixed(2)}, ${Mt[2][1].toFixed(2)}]</span> &nbsp; ` +
    `<span style="color:#60a5fa">k̂ [${Mt[0][2].toFixed(2)}, ${Mt[1][2].toFixed(2)}, ${Mt[2][2].toFixed(2)}]</span>`;

  if (S.atomIdx >= 0 && S.atomIdx < D.atoms.length) {
    const a = D.atoms[S.atomIdx];
    const [x, y, z] = a.p;
    const q = applyMat(Mt, a.p);
    $("atomEqBox").innerHTML =
      `r(0) = [${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}]<br>` +
      `r(t) = <span class="cv-txt">[${q[0].toFixed(2)}, ${q[1].toFixed(2)}, ${q[2].toFixed(2)}]</span>`;
  } else {
    $("atomEqBox").innerHTML = "Off";
  }
}

Object.keys(STYLES).forEach(k => {
  const b = document.createElement("button");
  b.textContent = k;
  b.onclick = () => { S.style = k; fullRedraw(); };
  $("styleSeg").appendChild(b);
});
$("tGrid").onclick = () => { S.showGrid = !S.showGrid; renderOverlay(); updateUI(); };
$("tBgGrid").onclick = () => { S.showBgGrid = !S.showBgGrid; renderOverlay(); updateUI(); };
$("tCube").onclick = () => { S.showCube = !S.showCube; renderOverlay(); updateUI(); };
$("tGhost").onclick = () => { S.showGhost = !S.showGhost; fullRedraw(); };
$("tDark").onclick = () => { S.dark = !S.dark; fullRedraw(); };
$("tReset").onclick = () => { v.zoomTo(); v.render(); };

$("mLin").onclick = () => { S.linearInterp = true; updateAtoms(); updateUI(); };
$("mGeod").onclick = () => { S.linearInterp = false; updateAtoms(); updateUI(); };
$("tLoop").onclick = () => { S.loop = !S.loop; updateUI(); };

$("playBtn").onclick = () => {
  if (S.playing) { S.playing = false; updateUI(); return; }
  if (S.t >= 1) S.t = 0;
  S.playing = true;
  startAnim();
};
$("scrub").oninput = () => {
  S.playing = false;
  S.t = $("scrub").value / 1000;
  updateAtoms();
  updateUI();
};

let lastTs = null, hold = 0;
function startAnim() { lastTs = null; hold = 0; requestAnimationFrame(tick); }
function tick(ts) {
  if (!S.playing) return;
  if (lastTs === null) lastTs = ts;
  const dt = Math.min((ts - lastTs) / 1000, 0.1);
  lastTs = ts;
  const dur = 2.4;

  if (hold > 0) {
    hold -= dt;
    if (hold <= 0) {
      if (S.loop) S.t = 0;
      else { S.playing = false; updateUI(); return; }
    }
  } else {
    S.t = Math.min(1, S.t + dt / dur);
    if (S.t >= 1) hold = 1.0;
  }
  updateAtoms();
  updateUI();
  requestAnimationFrame(tick);
}

buildMatrixEditor();
syncMatrixInputs();
v.zoomTo();
fullRedraw();
startAnim();
</script>
"""
    html = html.replace("__DATA__", json.dumps(data))
    components.html(html, height=690, scrolling=False)


# Header
st.markdown("""
<div class="ui top">
  <p class="title">Matrix transformation</p>
  <p class="subtitle">Upload an .xyz file to apply symmetry operations and 3×3 linear transformations in 3D space.</p>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns([3, 1.3], gap="large")
with c1:
    up = st.file_uploader("XYZ file", type=["xyz"], label_visibility="collapsed")
with c2:
    tol = st.slider(
        "Tolerance (Å)", 0.05, 1.0, 0.3, 0.05,
        help="How far atoms can be off and still count as symmetric."
    )

if up is None:
    st.markdown("""
<div class="ui empty"><b>No molecule yet</b><br>Drop an .xyz file above to get started.</div>
""", unsafe_allow_html=True)
    st.stop()

raw_xyz = up.getvalue().decode("utf-8", errors="ignore")

with st.spinner("Finding symmetry..."):
    if HAS_PMG:
        try:
            mol = Molecule.from_str(raw_xyz, fmt="xyz")
        except Exception as e:
            st.error(f"Couldn't read {up.name}. Error: {e}")
            st.stop()
        try:
            pga = PointGroupAnalyzer(mol, tolerance=tol, eigen_tolerance=max(0.01, tol * 0.08))
        except TypeError:
            pga = PointGroupAnalyzer(mol, tolerance=tol)
        pg = pga.sch_symbol
        cmol = pga.centered_mol
        coords = cmol.cart_coords
        species = [str(s.specie) for s in cmol]
        centered_xyz = cmol.to(fmt="xyz")
        matrices = [] if "*" in pg else clean_ops(pga.get_symmetry_operations())
    else:
        try:
            species, coords, centered_xyz = parse_xyz_fallback(raw_xyz)
        except Exception as e:
            st.error(f"Couldn't read {up.name}. Error: {e}")
            st.stop()
        pg, matrices = fallback_symmetry(species, coords, tol)

    if "*" in pg:
        sym_ops = build_linear_sym_ops(pg, coords)
        order_str = "∞"
    else:
        axes = get_axes(matrices, pg, coords, tol)
        normals = get_planes(matrices)
        types = label_planes(normals, pg, axes, coords, tol)
        order = sorted(range(len(types)), key=lambda i: PLANE_ORDER.index(types[i]))
        planes, seen = [], Counter()
        for i in order:
            t = types[i]
            seen[t] += 1
            name = sub(t) + (f" {seen[t]}" if types.count(t) > 1 else "")
            planes.append({"n": normals[i], "type": t, "name": name})
        sym_ops = build_sym_ops(matrices, axes, planes)
        order_str = str(len(matrices))

render_app(centered_xyz, coords, species, pg, sym_ops, order_str)