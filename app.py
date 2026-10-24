import json
import re
from collections import Counter
from fractions import Fraction

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

# wide layout
st.set_page_config(
    page_title="Molecular Space & Matrix Visualizer",
    page_icon="📐",
    layout="wide",
)

# clean styling matching your symmetry app
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {background:#ffffff;}
.block-container {padding-top:2.4rem; padding-bottom:3rem; max-width:1360px;}
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
    """Generate representative symmetry matrices for linear molecules (C*v and D*h)."""
    main = unit(np.linalg.svd(coords - coords.mean(0))[2][0])
    a = np.array([1.0, 0.0, 0.0]) if abs(main[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    p1 = unit(a - np.dot(a, main) * main)
    p2 = np.cross(main, p1)

    def clean(M):
        return np.where(np.abs(M) < 1e-6, 0.0, np.round(M, 4)).tolist()

    ops = [
        {
            "kind": "C", "u": main.tolist(), "a": float(np.pi / 2),
            "name": "C<sub>∞</sub> (90°)", "group": "Proper Rotations (det = +1)",
            "R": clean(rodrigues(main, np.pi / 2)),
            "desc": "Rotate 3D space by 90° around the C∞ molecular axis (det = +1)"
        },
        {
            "kind": "C", "u": main.tolist(), "a": float(np.pi),
            "name": "C<sub>∞</sub> (180°)", "group": "Proper Rotations (det = +1)",
            "R": clean(rodrigues(main, np.pi)),
            "desc": "Rotate 3D space by 180° around the C∞ molecular axis (det = +1)"
        },
        {
            "kind": "s", "n": p1.tolist(),
            "name": "σ<sub>v</sub> 1", "group": "Reflections (det = -1)",
            "R": clean(np.eye(3) - 2.0 * np.outer(p1, p1)),
            "desc": "Reflect 3D space across vertical plane σv 1 containing the molecular axis (det = -1)"
        },
        {
            "kind": "s", "n": p2.tolist(),
            "name": "σ<sub>v</sub> 2", "group": "Reflections (det = -1)",
            "R": clean(np.eye(3) - 2.0 * np.outer(p2, p2)),
            "desc": "Reflect 3D space across vertical plane σv 2 containing the molecular axis (det = -1)"
        },
    ]
    if pg == "D*h":
        ops.extend([
            {
                "kind": "C", "u": p1.tolist(), "a": float(np.pi),
                "name": "C<sub>2</sub>'", "group": "Proper Rotations (det = +1)",
                "R": clean(rodrigues(p1, np.pi)),
                "desc": "Rotate 3D space by 180° around a perpendicular C2 axis (det = +1)"
            },
            {
                "kind": "i", "name": "i (Inversion)", "group": "Inversion (det = -1)",
                "R": clean(-np.eye(3)),
                "desc": "Invert all of 3D space straight through the origin (R = -I, det = -1)"
            },
            {
                "kind": "s", "n": main.tolist(),
                "name": "σ<sub>h</sub>", "group": "Reflections (det = -1)",
                "R": clean(np.eye(3) - 2.0 * np.outer(main, main)),
                "desc": "Reflect 3D space across the horizontal mirror plane σh perpendicular to C∞ (det = -1)"
            },
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
                "kind": "s", "n": p["n"].tolist(), "name": p["name"], "group": "Reflections (det = -1)",
                "R": R_clean, "sort": (3, k, 0, 0, 0),
                "desc": f"Reflection across {p['name']}: space squishes flat onto the mirror plane (det = 0 at t=0.5) and flips inside-out (det = -1)"
            })
            continue

        if det < 0 and abs(tr + 3.0) < 0.08:
            out.append({
                "kind": "i", "name": "i (Inversion)", "group": "Inversion (det = -1)",
                "R": R_clean, "sort": (1, 0, 0, 0, 0),
                "desc": "Inversion through origin (R = -I): î, ĵ, k̂ and all grid lines pass through (0,0,0) and invert to -î, -ĵ, -k̂"
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

        deg = round(np.degrees(a))
        if letter == "C":
            ax_str = sub(axis_name(axes[j]["n"], axes[j].get("subtype"))) if j >= 0 else ""
            desc = f"Proper rotation by {deg}° about {ax_str}: the 3D grid rotates rigidly (det = +1) and every atom lands on an equivalent atom"
        else:
            desc = f"Improper rotation by {deg}° + reflection: space rotates then flips orientation across the perpendicular plane (det = -1)"

        out.append({
            "kind": letter, "u": u.tolist(), "a": a, "name": name,
            "R": R_clean, "desc": desc,
            "group": "Proper Rotations (det = +1)" if letter == "C" else "Improper Rotations (det = -1)",
            "sort": (0 if letter == "C" else 2, -n, len(primes), int(j >= 0 and axes[j]["n"] != n), j + k * 0.01)
        })

    out.sort(key=lambda o: o["sort"])
    for o in out:
        del o["sort"]
    return out


def atom_labels(mol):
    count, labs = Counter(), []
    for s in mol:
        el = str(s.specie)
        count[el] += 1
        labs.append(f"{el}{count[el]}")
    return labs


def render_app(cmol, pg, sym_ops, mol_name, order_str):
    xyz = cmol.to(fmt="xyz")
    coords = cmol.cart_coords
    size = float(max(np.linalg.norm(coords, axis=1).max(), 1.2))

    atoms = [
        {"p": c.tolist(), "el": str(s.specie), "lab": lab}
        for c, s, lab in zip(coords, cmol, atom_labels(cmol))
    ]

    data = {
        "xyz": xyz,
        "name": mol_name,
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

.layout {display:grid; grid-template-columns:1fr 420px; gap:16px; align-items:start;}
@media (max-width: 1050px) { .layout {grid-template-columns:1fr;} }

.card {background:#fff; border:1px solid #eceef1; border-radius:14px; overflow:hidden;
       box-shadow:0 1px 2px rgba(16,24,40,.04); margin-bottom:12px;}
.pad {padding:15px 17px;}

.bar {display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:space-between;
      padding:10px 12px; border-bottom:1px solid #f1f2f4; background:#fff;}
.seg {display:inline-flex; background:#f3f4f6; border-radius:9px; padding:3px;}
.seg button {border:0; background:transparent; padding:5px 11px; font-size:12px; border-radius:7px;
             cursor:pointer; color:#4b5563; font-weight:500;}
.seg button.on {background:#fff; color:#111827; box-shadow:0 1px 2px rgba(0,0,0,.1); font-weight:600;}

.group {display:inline-flex; gap:6px; flex-wrap:wrap; align-items:center;}
.chip {border:1px solid #e5e7eb; background:#fff; border-radius:999px; padding:4px 10px; font-size:12px;
       cursor:pointer; color:#374151; display:inline-flex; align-items:center; gap:5px; font-weight:500;}
.chip:hover {border-color:#cbd0d8;}
.chip.off {color:#9ca3af; background:#fafafa;}
.chip .d {width:8px; height:8px; border-radius:50%;}
.chip.off .d {opacity:.25;}

#vwrap {position:relative; width:100%; height:560px; background:#0b101b; overflow:hidden;}
#vwrap.light {background:#ffffff;}
#v {width:100%; height:100%; position:relative;}
#gridCv {position:absolute; inset:0; width:100%; height:100%; pointer-events:none; z-index:4;}
#ovLabs {position:absolute; inset:0; pointer-events:none; overflow:hidden; z-index:5;}

.btag {position:absolute; left:0; top:0; font-size:12px; font-weight:700; color:#fff; padding:1px 6px;
       border-radius:6px; white-space:nowrap; will-change:transform; box-shadow:0 1px 3px rgba(0,0,0,.3);}

.hud {position:absolute; left:12px; top:12px; z-index:6; padding:7px 11px; border-radius:10px; font-size:12px;
      background:rgba(15, 23, 42, 0.78); color:#f8fafc; backdrop-filter:blur(6px);
      border:1px solid rgba(255,255,255,.12); pointer-events:none; line-height:1.45;}
#vwrap.light .hud {background:rgba(255,255,255,.9); color:#111827; border-color:#e5e7eb;}

.anim {padding:12px 14px; border-top:1px solid #f1f2f4; background:#fafbfc;}
.anim-row {display:flex; flex-wrap:wrap; gap:8px; align-items:center;}
.play {border:0; background:#111827; color:#fff; border-radius:9px; padding:6px 16px; font-size:13px;
       font-weight:600; cursor:pointer; min-width:76px;}
.anim input[type=range] {flex:1; min-width:130px; accent-color:#111827; cursor:pointer;}
.cap {font-size:12.5px; color:#374151; margin-top:8px; line-height:1.45;}

.sec-title {font-size:11.5px; font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:.04em; margin:0 0 8px 0;}

/* Custom HTML Dropdown for Symmetry Ops */
.dd {position:relative; width:100%; margin-bottom:10px;}
.dd-btn {display:flex; width:100%; align-items:center; justify-content:space-between; gap:10px; font-size:13px;
         padding:7px 11px; border:1px solid #d1d5db; border-radius:9px; background:#fff; color:#111827;
         cursor:pointer; font-weight:500;}
.dd-btn:hover {border-color:#9ca3af;}
.dd-menu {display:none; position:absolute; left:0; top:calc(100% + 4px); z-index:30; width:100%; max-height:260px;
          overflow:auto; background:#fff; border:1px solid #e5e7eb; border-radius:10px;
          box-shadow:0 10px 25px rgba(16,24,40,.14); padding:4px;}
.dd.open .dd-menu {display:block;}
.dd-group {font-size:11px; color:#6b7280; font-weight:700; padding:7px 10px 3px; text-transform:uppercase;}
.dd-item {display:block; width:100%; text-align:left; border:0; background:none; padding:6px 10px; font-size:13px;
          border-radius:7px; cursor:pointer; color:#111827;}
.dd-item:hover {background:#f3f4f6;}
.dd-item.sel {background:#eef2ff; font-weight:600; color:#1d4ed8;}
sub, sup {font-size:.72em; line-height:0;}
.ss {display:inline-flex; flex-direction:column; vertical-align:middle; font-size:.7em; line-height:1; margin-left:1px;}
.ss sup, .ss sub {font-size:1em; line-height:1; vertical-align:baseline;}

/* 3x3 Matrix Bracket Editor */
.mat-box {display:flex; align-items:center; justify-content:center; gap:5px; margin:4px 0 10px 0;}
.bkt {width:8px; align-self:stretch; border:2px solid #111827;}
.bkt.l {border-right:0; border-radius:4px 0 0 4px;}
.bkt.r {border-left:0; border-radius:0 4px 4px 0;}
.mgrid {display:grid; grid-template-columns:repeat(3, 74px); gap:5px 7px; padding:3px;}
.mhdr {display:grid; grid-template-columns:repeat(3, 74px); gap:0 7px; text-align:center; font-size:11.5px; font-weight:700; margin-bottom:2px;}

.minp {width:100%; padding:5px 4px; font-size:13px; font-weight:600; text-align:center; border-radius:7px;
       border:1.5px solid #e5e7eb; font-variant-numeric:tabular-nums; outline:none;}
.minp:focus {border-color:#2563eb;}
.minp.ci {color:#15803d; border-color:#bbf7d0; background:#f0fdf4;}
.minp.cj {color:#b91c1c; border-color:#fecaca; background:#fef2f2;}
.minp.ck {color:#1d4ed8; border-color:#bfdbfe; background:#eff6ff;}

.ci-txt {color:#16a34a; font-weight:700;}
.cj-txt {color:#dc2626; font-weight:700;}
.ck-txt {color:#2563eb; font-weight:700;}
.cv-txt {color:#d97706; font-weight:700;}

.presets {display:flex; flex-wrap:wrap; gap:5px; margin-top:6px;}
.pbtn {border:1px solid #e5e7eb; background:#f9fafb; border-radius:7px; padding:4px 8px; font-size:11.5px;
       color:#374151; cursor:pointer; font-weight:500;}
.pbtn:hover {background:#f3f4f6; border-color:#cbd0d8; color:#111827;}

.stats {display:grid; grid-template-columns:1fr 1fr 1fr; gap:7px; margin-bottom:9px;}
.stat {border:1px solid #eceef1; border-radius:9px; padding:8px 10px; background:#fafbfc;}
.stat .k {font-size:11px; color:#6b7280; font-weight:500;}
.stat .v {font-size:15.5px; font-weight:700; margin-top:2px; font-variant-numeric:tabular-nums;}

.badge {display:inline-block; padding:3px 9px; border-radius:999px; font-size:11.5px; font-weight:600; margin-bottom:6px;}
.badge.ok {background:#dcfce7; color:#166534;}
.badge.warn {background:#fee2e2; color:#991b1b;}
.badge.zero {background:#fef3c7; color:#92400e;}

.eq-box {background:#f8fafc; border:1px solid #e5e7eb; border-radius:9px; padding:9px 11px;
         font-size:12px; line-height:1.5; color:#1f2937; font-variant-numeric:tabular-nums;}
select.small {font-size:12px; padding:4px 7px; border:1px solid #e5e7eb; border-radius:8px; background:#fff;}
</style>

<div class="layout">
  <!-- LEFT: 3Dmol + 3Blue1Brown Warping 3D Coordinate Space -->
  <div>
    <div class="card">
      <div class="bar">
        <div class="seg" id="styleSeg"></div>
        <div class="group">
          <button class="chip" id="tGrid"><span class="d" style="background:#38bdf8"></span>3b1b Space Grid</button>
          <button class="chip" id="tBgGrid"><span class="d" style="background:#94a3b8"></span>Base Grid</button>
          <button class="chip" id="tCube"><span class="d" style="background:#22c55e"></span>Unit Cube (det)</button>
          <button class="chip" id="tGhost"><span class="d" style="background:#9ca3af"></span>Ghost Molecule</button>
          <button class="chip" id="tDark">Dark (3b1b)</button>
          <button class="chip" id="tReset">Reset View</button>
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
          <input type="range" id="scrub" min="0" max="1000" value="0" aria-label="Transformation progress">
          <div class="seg" id="interpSeg" title="How space interpolates from I to M">
            <button id="mLin" class="on">Linear (1−t)I + tM</button>
            <button id="mGeod">Rigid Path</button>
          </div>
          <button class="chip" id="tLoop">Loop</button>
        </div>
        <div class="cap" id="capTxt"></div>
      </div>
    </div>
  </div>

  <!-- RIGHT: Symmetry Operations vs. General 3Blue1Brown Matrix Editor -->
  <div>
    <div class="card pad">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <p class="sec-title" style="margin:0">1. Symmetry Matrices of <span id="molTitle"></span></p>
        <span style="font-size:12px; font-weight:700; color:#111827;" id="pgBadge"></span>
      </div>
      <div class="dd" id="symDD"></div>

      <p class="sec-title" style="margin-top:8px; margin-bottom:4px;">2. Or Apply Any 3Blue1Brown 3×3 Matrix M</p>
      <div style="display:flex; flex-direction:column; align-items:center;">
        <div class="mhdr">
          <span class="ci-txt">î (col 1)</span>
          <span class="cj-txt">ĵ (col 2)</span>
          <span class="ck-txt">k̂ (col 3)</span>
        </div>
        <div class="mat-box">
          <div class="bkt l"></div>
          <div class="mgrid" id="matInputs"></div>
          <div class="bkt r"></div>
        </div>
      </div>

      <p class="sec-title" style="margin-top:4px; margin-bottom:4px;">3b1b Linear Algebra Presets (Non-Symmetry Warps)</p>
      <div class="presets" id="presetBox"></div>
    </div>

    <div class="card pad">
      <p class="sec-title">Live Space & Symmetry Verification</p>
      <div class="stats">
        <div class="stat">
          <div class="k">Target det(M)</div>
          <div class="v" id="detTarget">1.00</div>
        </div>
        <div class="stat">
          <div class="k">Live det M(t)</div>
          <div class="v" id="detLive">1.00</div>
        </div>
        <div class="stat">
          <div class="k">Symmetry RMSD</div>
          <div class="v" id="rmsdVal">0.00 Å</div>
        </div>
      </div>
      <div id="symVerdict"></div>
      <div class="eq-box" id="liveMatBox"></div>
    </div>

    <div class="card pad">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <p class="sec-title" style="margin:0">Track Atom Vector <span class="cv-txt">r⃗ = xî + yĵ + zk̂</span></p>
        <select id="atomSel" class="small"></select>
      </div>
      <div class="eq-box" id="atomEqBox"></div>
    </div>
  </div>
</div>

<script>
const D = __DATA__;
const $ = id => document.getElementById(id);

$("molTitle").textContent = D.name;
$("pgBadge").innerHTML = `${D.pg} (h = ${D.orderStr})`;

const STYLES = {
  "Ball & stick": {stick: {radius: 0.13}, sphere: {scale: 0.24}},
  "Sticks": {stick: {radius: 0.17}},
  "Spacefill": {sphere: {scale: 0.85}}
};

const PRESETS_3B1B = [
  {name: "3b1b Shear (1:01 style)", M: [[1, 0.85, 0], [0.45, 1, 0], [0, 0, 1]],
   desc: "3Blue1Brown Shear: grid lines stay parallel and evenly spaced, skewing the molecule off its ghost atoms!"},
  {name: "Stretch Z (1.6×)", M: [[1, 0, 0], [0, 1, 0], [0, 0, 1.6]],
   desc: "Anisotropic scaling along k̂: stretches the molecule vertically (det = 1.60), changing bond lengths."},
  {name: "Collapse to XY Plane (det=0)", M: [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
   desc: "Singular Matrix (det = 0): squishes all of 3D space and the 3D molecule flat onto the z = 0 plane!"},
  {name: "Collapse to Line (Rank 1)", M: [[1, 0.5, 0.5], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]],
   desc: "Rank-1 Matrix (det = 0): collapses the entire 3D grid and every atom onto a single 1D line!"},
  {name: "Identity I", M: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
   desc: "Identity matrix I: basis vectors stay at î=[1,0,0], ĵ=[0,1,0], k̂=[0,0,1]."}
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
  M: hasSymOps ? D.symOps[0].R.map(r => [...r]) : PRESETS_3B1B[0].M.map(r => [...r]),
  customDesc: hasSymOps ? "" : PRESETS_3B1B[0].desc,
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
  const g = {color: S.dark ? "#64748b" : "#94a3b8", opacity: 0.28};
  if (!S.showGhost) { ghost.setStyle({}, {}); return; }
  if (S.style === "Spacefill") ghost.setStyle({}, {sphere: {...g, scale: 0.86}});
  else if (S.style === "Sticks") ghost.setStyle({}, {stick: {...g, radius: 0.14}});
  else ghost.setStyle({}, {sphere: {...g, scale: 0.22}, stick: {...g, radius: 0.09}});
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
  root.innerHTML = '<button type="button" class="dd-btn"><span class="dd-val"></span><span>▾</span></button><div class="dd-menu"></div>';
  const btn = root.querySelector(".dd-btn"), menu = root.querySelector(".dd-menu"), valSpan = root.querySelector(".dd-val");

  if (!hasSymOps) {
    const h = document.createElement("div");
    h.className = "dd-group";
    h.textContent = "Only E (C1 molecule — use 3b1b presets below)";
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
          S.customDesc = "";
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
        valSpan.innerHTML = `Symmetry Op: <b>${D.symOps[S.symIdx].name}</b>`;
      } else if (!hasSymOps) {
        valSpan.innerHTML = `<i>No non-trivial symmetry ops (C₁) — using 3×3 Matrix</i>`;
      } else {
        valSpan.innerHTML = `<i>Custom / Non-Symmetry 3×3 Matrix M</i>`;
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
          S.customDesc = "Custom 3×3 matrix transformation: watch how the 3D coordinate grid and molecule warp together.";
          updateAtoms();
          updateUI();
        }
      };
      grid.appendChild(inp);
    }
  }

  const pBox = $("presetBox");
  PRESETS_3B1B.forEach(p => {
    const b = document.createElement("button");
    b.className = "pbtn";
    b.textContent = p.name;
    b.onclick = () => {
      S.M = p.M.map(r => [...r]);
      S.symIdx = -1;
      S.customDesc = p.desc;
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
    opt.textContent = `${a.lab} (${a.p[0].toFixed(2)}, ${a.p[1].toFixed(2)}, ${a.p[2].toFixed(2)})`;
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
  $("tDark").textContent = S.dark ? "Dark (3b1b)" : "Light";
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

  const isSym = rmsd < 0.08 && Math.abs(Math.abs(dtTarget) - 1.0) < 0.05;
  if (Math.abs(dtTarget) < 0.04) {
    $("symVerdict").innerHTML = `<span class="badge zero">Singular Matrix (det = 0): Space collapses into lower dimension!</span>`;
  } else if (isSym) {
    $("symVerdict").innerHTML = `<span class="badge ok">✓ True Symmetry Matrix of ${D.name}: Grid warps & atoms land on identical atoms!</span>`;
  } else {
    $("symVerdict").innerHTML = `<span class="badge warn">✗ Non-Symmetry Matrix: Space warps/stretches the molecule off its ghost (RMSD = ${rmsd.toFixed(2)} Å)</span>`;
  }

  $("hudBox").innerHTML =
    `<b>Live Basis Columns of M(t) (t = ${S.t.toFixed(2)}):</b><br>` +
    `<span style="color:#4ade80">î(t) = [${Mt[0][0].toFixed(2)}, ${Mt[1][0].toFixed(2)}, ${Mt[2][0].toFixed(2)}]</span> &nbsp;|&nbsp; ` +
    `<span style="color:#f87171">ĵ(t) = [${Mt[0][1].toFixed(2)}, ${Mt[1][1].toFixed(2)}, ${Mt[2][1].toFixed(2)}]</span> &nbsp;|&nbsp; ` +
    `<span style="color:#60a5fa">k̂(t) = [${Mt[0][2].toFixed(2)}, ${Mt[1][2].toFixed(2)}, ${Mt[2][2].toFixed(2)}]</span>`;

  $("liveMatBox").innerHTML =
    `<b>How the 3Blue1Brown Grid Warps:</b><br>` +
    (S.linearInterp
      ? `Using pure linear interpolation <b>M(t) = (1 − t)I + tM</b>. Notice how for reflections or inversions, the grid lines pass straight through one another (det M(t) crosses 0) as space turns inside-out!`
      : `Using rigid rotation/reflection path. Switch to <b>Linear (1−t)I + tM</b> on the bottom bar to see the pure 3Blue1Brown linear grid morph!`);

  if (S.symIdx >= 0 && D.symOps[S.symIdx]) {
    $("capTxt").innerHTML = D.symOps[S.symIdx].desc;
  } else {
    $("capTxt").innerHTML = S.customDesc;
  }

  if (S.atomIdx >= 0 && S.atomIdx < D.atoms.length) {
    const a = D.atoms[S.atomIdx];
    const [x, y, z] = a.p;
    const q = applyMat(Mt, a.p);
    $("atomEqBox").innerHTML =
      `<b>M(t) · <span class="cv-txt">${a.lab}</span></b> = ` +
      `(${x.toFixed(2)})<span class="ci-txt">î(t)</span> + (${y.toFixed(2)})<span class="cj-txt">ĵ(t)</span> + (${z.toFixed(2)})<span class="ck-txt">k̂(t)</span><br>` +
      `Start: [${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}] &nbsp;→&nbsp; <b class="cv-txt">Now: [${q[0].toFixed(2)}, ${q[1].toFixed(2)}, ${q[2].toFixed(2)}]</b>`;
  } else {
    $("atomEqBox").innerHTML = "Select an atom above to see its tip-to-tail linear combination xî(t) + yĵ(t) + zk̂(t).";
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
  const dur = 2.5;

  if (hold > 0) {
    hold -= dt;
    if (hold <= 0) {
      if (S.loop) S.t = 0;
      else { S.playing = false; updateUI(); return; }
    }
  } else {
    S.t = Math.min(1, S.t + dt / dur);
    if (S.t >= 1) hold = 1.1;
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
    components.html(html, height=730, scrolling=False)


# Header matching your symmetry app
st.markdown("""
<div class="ui top">
  <p class="title">Molecular Space & Matrix Visualizer</p>
  <p class="subtitle">Upload any .xyz file to visualize its symmetry matrices and 3Blue1Brown 3×3 linear transformations in 3D coordinate space.</p>
</div>
""", unsafe_allow_html=True)

# File uploader on the left, tolerance on the right
c1, c2 = st.columns([3, 1.3], gap="large")
with c1:
    up = st.file_uploader("XYZ file", type=["xyz"], label_visibility="collapsed")
with c2:
    tol = st.slider(
        "Tolerance (Å)", 0.05, 1.0, 0.3, 0.05,
        help="How far atoms can be off and still count as symmetric. "
             "Raise it if a symmetric molecule comes out as C1."
    )

# Empty state until user drops an .xyz file
if up is None:
    st.markdown("""
<div class="ui empty"><b>No molecule yet</b><br>Drop an .xyz file above to get started.</div>
""", unsafe_allow_html=True)
    st.stop()

try:
    mol = Molecule.from_str(up.getvalue().decode("utf-8", errors="ignore"), fmt="xyz")
    mol_name = up.name.rsplit(".", 1)[0]
except Exception as e:
    st.error(
        f"Couldn't read {up.name}. Check the file is a normal xyz "
        f"(atom count, comment line, then element x y z). Error: {e}"
    )
    st.stop()

with st.spinner("Finding symmetry matrices..."):
    try:
        pga = PointGroupAnalyzer(mol, tolerance=tol, eigen_tolerance=max(0.01, tol * 0.08))
    except TypeError:
        pga = PointGroupAnalyzer(mol, tolerance=tol)
    pg = pga.sch_symbol
    cmol = pga.centered_mol

    if "*" in pg:
        sym_ops = build_linear_sym_ops(pg, cmol.cart_coords)
        order_str = "∞"
    else:
        matrices = clean_ops(pga.get_symmetry_operations())
        axes = get_axes(matrices, pg, cmol.cart_coords, tol)
        normals = get_planes(matrices)
        types = label_planes(normals, pg, axes, cmol.cart_coords, tol)
        order = sorted(range(len(types)), key=lambda i: PLANE_ORDER.index(types[i]))
        planes, seen = [], Counter()
        for i in order:
            t = types[i]
            seen[t] += 1
            name = sub(t) + (f" {seen[t]}" if types.count(t) > 1 else "")
            planes.append({"n": normals[i], "type": t, "name": name})
        sym_ops = build_sym_ops(matrices, axes, planes)
        order_str = str(len(matrices))

render_app(cmol, pg, sym_ops, mol_name, order_str)