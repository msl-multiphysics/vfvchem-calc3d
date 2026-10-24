"""obj geometry groups and dual control volumes of a VFVChem domain."""

from __future__ import annotations

import numpy as np


def read_obj_groups(path: str) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    """Read bnd*/itf* and dom* polygons from a VFVChem obj file.

    Returns 2D groups (boundaries and interfaces) and 3D groups (domains) as
    name -> list of planar polygons, each an (n_vert, 3) array in winding order.
    """
    verts: list[tuple[float, float, float]] = []
    faces: dict[str, list[tuple[int, list[int]]]] = {}
    cur: str | None = None
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.split("#")[0].strip()
            if not line:
                continue
            tok = line.split()
            if tok[0] == "v":
                if not 4 <= len(tok) <= 5:
                    raise ValueError(f"{path}: line {line_no}: vertex must have x, y, and z coordinates")
                verts.append((float(tok[1]), float(tok[2]), float(tok[3])))
            elif tok[0] == "g":
                if len(tok) != 2:
                    raise ValueError(f"{path}: line {line_no}: group must have one name")
                if not tok[1].startswith(("bnd", "itf", "dom")):
                    raise ValueError(
                        f"{path}: line {line_no}: group name {tok[1]!r} must begin with 'bnd', 'itf', or 'dom'"
                    )
                # repeated declarations continue the same group
                cur = tok[1]
                faces.setdefault(cur, [])
            elif tok[0] == "f":
                if cur is None:
                    raise ValueError(f"{path}: line {line_no}: face is not in a group")
                # keep the vertex index of v/vt/vn records
                vids = [int(t.split("/")[0]) - 1 for t in tok[1:]]
                if len(vids) < 3:
                    raise ValueError(f"{path}: line {line_no}: face must contain at least three vertices")
                faces[cur].append((line_no, vids))
            else:
                raise ValueError(f"{path}: line {line_no}: unsupported record {tok[0]!r}")

    pos = np.asarray(verts, dtype=float)
    grp2d: dict[str, list[np.ndarray]] = {}
    grp3d: dict[str, list[np.ndarray]] = {}
    for name, recs in faces.items():
        for line_no, vids in recs:
            if min(vids) < 0 or max(vids) >= len(pos):
                raise ValueError(f"{path}: line {line_no}: vertex index is out of range")
        polys = [pos[vids] for _, vids in recs]
        if polys:
            (grp3d if name.startswith("dom") else grp2d)[name] = polys
    return grp2d, grp3d


def bounds_groups(
    x_min: float, y_min: float, z_min: float, x_max: float, y_max: float, z_max: float
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    """Groups of ``Variables::new_bounds``: bnd_l, bnd_r, bnd_f, bnd_k, bnd_b, bnd_t, and dom."""
    if x_min >= x_max:
        raise ValueError(f"Invalid x bounds: {x_min} >= {x_max}")
    if y_min >= y_max:
        raise ValueError(f"Invalid y bounds: {y_min} >= {y_max}")
    if z_min >= z_max:
        raise ValueError(f"Invalid z bounds: {z_min} >= {z_max}")

    # box vertices; vertex id = i + 2 * j + 4 * k for (x, y, z) = (x_i, y_j, z_k)
    pos = np.asarray(
        [(x, y, z) for z in (z_min, z_max) for y in (y_min, y_max) for x in (x_min, x_max)], dtype=float
    )

    # faces with outward normals
    faces = {
        "bnd_l": [0, 4, 6, 2],  # left (x_min)
        "bnd_r": [1, 3, 7, 5],  # right (x_max)
        "bnd_f": [0, 1, 5, 4],  # front (y_min)
        "bnd_k": [2, 6, 7, 3],  # back (y_max)
        "bnd_b": [0, 2, 3, 1],  # bottom (z_min)
        "bnd_t": [4, 5, 7, 6],  # top (z_max)
    }
    grp2d = {name: [pos[vids]] for name, vids in faces.items()}
    grp3d = {"dom": [pos[vids] for vids in faces.values()]}
    return grp2d, grp3d


def on_polygons(pts: np.ndarray, polys: list[np.ndarray], tol: float) -> np.ndarray:
    """Boolean mask of points within ``tol`` of any planar polygon."""
    mask = np.zeros(len(pts), dtype=bool)
    for poly in polys:
        # Newell normal and in-plane frame (e1, e2)
        nrm = np.sum(np.cross(poly, np.roll(poly, -1, axis=0)), axis=0)
        nrm /= np.linalg.norm(nrm)
        e1 = poly[1] - poly[0]
        e1 -= np.dot(e1, nrm) * nrm
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nrm, e1)

        # candidates: inside the bounding box and close to the plane
        near = np.all((pts >= poly.min(axis=0) - tol) & (pts <= poly.max(axis=0) + tol), axis=1)
        near &= np.abs((pts - poly[0]) @ nrm) <= tol
        if not near.any():
            continue
        rel = pts[near] - poly[0]
        px, py = rel @ e1, rel @ e2
        q = (poly - poly[0]) @ np.stack([e1, e2], axis=1)
        x0, y0 = q[:, 0][None, :], q[:, 1][None, :]
        x1, y1 = np.roll(q[:, 0], -1)[None, :], np.roll(q[:, 1], -1)[None, :]
        px, py = px[:, None], py[:, None]

        # even-odd crossing test
        crosses = ((y0 > py) != (y1 > py)) & (px < x0 + (py - y0) * (x1 - x0) / np.where(y1 != y0, y1 - y0, 1.0))
        inside = np.count_nonzero(crosses, axis=1) % 2 == 1

        # points on the polygon edges, e.g., on the shared edges of merged coplanar polygons
        ex, ey = x1 - x0, y1 - y0
        t = np.clip(((px - x0) * ex + (py - y0) * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        d2 = (x0 + t * ex - px) ** 2 + (y0 + t * ey - py) ** 2
        inside |= np.min(d2, axis=1) <= tol * tol

        idx = np.flatnonzero(near)
        mask[idx[inside]] = True
    return mask


def circumcircle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Centers (k, 3) of the circles through three points."""
    u, v = b - a, c - a
    w = np.cross(u, v)
    num = np.sum(u * u, axis=1)[:, None] * np.cross(v, w) + np.sum(v * v, axis=1)[:, None] * np.cross(w, u)
    return a + (0.5 / np.sum(w * w, axis=1))[:, None] * num


def circumsphere3(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Centers (k, 3) of the spheres through four points."""
    u, v, w = b - a, c - a, d - a
    num = (
        np.sum(u * u, axis=1)[:, None] * np.cross(v, w)
        + np.sum(v * v, axis=1)[:, None] * np.cross(w, u)
        + np.sum(w * w, axis=1)[:, None] * np.cross(u, v)
    )
    return a + (0.5 / np.sum(u * np.cross(v, w), axis=1))[:, None] * num


class Dual3D:
    """Voronoi control volumes of one domain; port of ``Tetrahedron::update_tet``.

    Vertices are ordered points first, then nodes, as in the VTU files.
    Edges are stored once as (a, b) with a < b; edge normals point from a to b.
    """

    def __init__(self, pos: np.ndarray, tet: np.ndarray) -> None:
        pos = np.asarray(pos, dtype=float)[:, :3]
        tet = np.asarray(tet, dtype=int)
        n = len(pos)
        if tet.ndim != 2 or tet.shape[1] != 4 or len(tet) < 1:
            raise ValueError(f"Expected tetrahedra of shape (n, 4), got {tet.shape}")
        self.pos = pos
        self.tet = tet
        self.num_tot = n

        # boundary faces are faces of exactly one tetrahedron
        f_v = np.concatenate([tet[:, [(k + 1) % 4, (k + 2) % 4, (k + 3) % 4]] for k in range(4)])
        f_o = np.concatenate([tet[:, k] for k in range(4)])
        _, inv, count = np.unique(np.sort(f_v, axis=1), axis=0, return_inverse=True, return_counts=True)
        is_bnd = count[np.asarray(inv).ravel()] == 1
        bnd_tri = f_v[is_bnd]

        # counterclockwise seen from outside; the opposite vertex is inside
        p0, p1, p2 = pos[bnd_tri[:, 0]], pos[bnd_tri[:, 1]], pos[bnd_tri[:, 2]]
        flip = np.sum((pos[f_o[is_bnd]] - p0) * np.cross(p1 - p0, p2 - p0), axis=1) > 0.0
        bnd_tri[flip] = bnd_tri[flip][:, [0, 2, 1]]
        self.bnd_tri = bnd_tri

        # nodes are the vertices on boundary faces and are stored after the points
        is_node = np.zeros(n, dtype=bool)
        is_node[bnd_tri.ravel()] = True
        self.num_pts = n - int(np.count_nonzero(is_node))
        if is_node[: self.num_pts].any() or not is_node[self.num_pts :].all():
            raise ValueError("VTU vertices are not ordered as points followed by nodes")

        # Voronoi faces
        # the face piece of edge (i, j) on the side of face (i, j, a) is the right triangle (m, c_f, c_t)
        # m is the edge midpoint, c_f the face circumcenter, and c_t the tetrahedron circumcenter
        # its signed area is 0.5 * s1 * s2 with s1 = (c_f - m) . u toward a and s2 = (c_t - c_f) . n toward b
        c_t = circumsphere3(pos[tet[:, 0]], pos[tet[:, 1]], pos[tet[:, 2]], pos[tet[:, 3]])

        # flat tetrahedra of cocircular vertices have no circumsphere and their face pieces cancel exactly;
        # the 9-decimal VTU coordinates spoil the cancellation, so use the mean face circumcenter instead
        q = pos[tet]
        det = np.sum((q[:, 1] - q[:, 0]) * np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]), axis=1)
        len_max = np.max(np.linalg.norm(q[:, [1, 2, 3, 2, 3, 3]] - q[:, [0, 0, 0, 1, 1, 2]], axis=2), axis=1)
        flat = np.abs(det) <= 1e-9 * len_max**3
        if flat.any():
            q = q[flat]
            c_t[flat] = np.mean([circumcircle3(q[:, (k + 1) % 4], q[:, (k + 2) % 4], q[:, (k + 3) % 4]) for k in range(4)], axis=0)

        loc = [(i, j, a, b) for i, j, k, l in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)]
               for a, b in [(k, l), (l, k)]]
        v_i = np.concatenate([tet[:, i] for i, _, _, _ in loc])
        v_j = np.concatenate([tet[:, j] for _, j, _, _ in loc])
        p_i, p_j = pos[v_i], pos[v_j]
        p_a = pos[np.concatenate([tet[:, a] for _, _, a, _ in loc])]
        p_b = pos[np.concatenate([tet[:, b] for _, _, _, b in loc])]
        p_t = np.tile(c_t, (len(loc), 1))

        dist = np.linalg.norm(p_j - p_i, axis=1)
        e = (p_j - p_i) / dist[:, None]
        p_m = 0.5 * (p_i + p_j)
        p_f = circumcircle3(p_i, p_j, p_a)
        u = (p_a - p_m) - np.sum((p_a - p_m) * e, axis=1)[:, None] * e
        u /= np.linalg.norm(u, axis=1)[:, None]
        nrm = np.cross(e, u)
        nrm *= np.where(np.sum((p_b - p_m) * nrm, axis=1) < 0.0, -1.0, 1.0)[:, None]
        piece = 0.5 * np.sum((p_f - p_m) * u, axis=1) * np.sum((p_t - p_f) * nrm, axis=1)
        cent = (p_m + p_f + p_t) / 3.0
        self.piece_mid = p_m
        self.piece_face = p_f
        self.piece_circ = p_t

        # accumulate face pieces per edge
        edge, inv = np.unique(np.stack([np.minimum(v_i, v_j), np.maximum(v_i, v_j)], axis=1), axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        num_edge = len(edge)
        face = np.bincount(inv, piece, num_edge)
        skew_sum = np.stack([np.bincount(inv, piece * (cent[:, d] - p_m[:, d]), num_edge) for d in range(3)], axis=1)

        # pyramids from vertices i and j over each piece; signed volume piece * dist / 6, centroid 3/4 to the piece center
        vol = piece * dist / 6.0
        cent_sum = np.zeros((n, 3))
        np.add.at(cent_sum, v_i, (0.75 * vol)[:, None] * (cent - p_i))
        np.add.at(cent_sum, v_j, (0.75 * vol)[:, None] * (cent - p_j))

        # edge, face, and control volume data
        a, b = edge[:, 0], edge[:, 1]
        d = pos[b] - pos[a]
        dist = np.linalg.norm(d, axis=1)
        is_face = np.abs(face) > 1e-12 * dist * dist
        safe = np.where(is_face, face, 1.0)
        self.edge = edge
        self.face = face
        self.dist = dist
        self.norm = d / dist[:, None]
        self.skew = np.where(is_face[:, None], skew_sum / safe[:, None], 0.0)
        self.cvol_vol = np.bincount(a, face * dist / 6.0, n) + np.bincount(b, face * dist / 6.0, n)
        self.cvol_skew = cent_sum / self.cvol_vol[:, None]

        # boundary sub-faces
        # the sub-face of vertex i in boundary face (i, j, k) is made of the signed triangles (i, m_ij, c) and (i, m_ik, c)
        # each has area 0.25 * |ij| * s_ij, where s_ij = (c - m_ij) . u_ij and u_ij points from edge ij toward k in the plane
        # centroids are kept in barycentric coordinates (w_i, w_j, w_k) for interpolation
        q = [pos[bnd_tri[:, r]] for r in range(3)]
        tri_n = np.cross(q[1] - q[0], q[2] - q[0])
        tri_n /= np.linalg.norm(tri_n, axis=1)[:, None]
        circ = circumcircle3(q[0], q[1], q[2])
        self.bnd_circ = circ

        # barycentric coordinates of the circumcenter; lam_i is proportional to a^2 (b^2 + c^2 - a^2) with a opposite i
        sq = [np.sum((q[(r + 2) % 3] - q[(r + 1) % 3]) ** 2, axis=1) for r in range(3)]
        lam = np.stack([sq[r] * (sq[(r + 1) % 3] + sq[(r + 2) % 3] - sq[r]) for r in range(3)], axis=1)
        lam /= np.sum(lam, axis=1)[:, None]

        bnd_v, bnd_j, bnd_k, bnd_area, bnd_wgt = [], [], [], [], []
        for r in range(3):
            pi, pj, pk = q[r], q[(r + 1) % 3], q[(r + 2) % 3]
            area = []
            for pa, pb in ((pj, pk), (pk, pj)):
                m = 0.5 * (pi + pa)
                ed = pa - pi
                ln = np.linalg.norm(ed, axis=1)
                uu = (pb - m) - (np.sum((pb - m) * ed, axis=1) / (ln * ln))[:, None] * ed
                area.append(0.25 * ln * np.sum((circ - m) * uu, axis=1) / np.linalg.norm(uu, axis=1))
            area_sum = area[0] + area[1]

            # centroid weights; (i + m_ij + c) / 3 and (i + m_ik + c) / 3 in barycentric coordinates (i, j, k)
            l_i, l_j, l_k = lam[:, r], lam[:, (r + 1) % 3], lam[:, (r + 2) % 3]
            cent_j = np.stack([(1.5 + l_i) / 3.0, (0.5 + l_j) / 3.0, l_k / 3.0], axis=1)
            cent_k = np.stack([(1.5 + l_i) / 3.0, l_j / 3.0, (0.5 + l_k) / 3.0], axis=1)
            ok = np.abs(area_sum) > 1e-12 * np.linalg.norm(pj - pi, axis=1) * np.linalg.norm(pk - pi, axis=1)
            wgt = (area[0][:, None] * cent_j + area[1][:, None] * cent_k) / np.where(ok, area_sum, 1.0)[:, None]
            wgt[~ok] = (1.0, 0.0, 0.0)

            bnd_v.append(bnd_tri[:, r])
            bnd_j.append(bnd_tri[:, (r + 1) % 3])
            bnd_k.append(bnd_tri[:, (r + 2) % 3])
            bnd_area.append(area_sum)
            bnd_wgt.append(wgt)
        self.bnd_v = np.concatenate(bnd_v)
        self.bnd_j = np.concatenate(bnd_j)
        self.bnd_k = np.concatenate(bnd_k)
        self.bnd_area = np.concatenate(bnd_area)
        self.bnd_norm = np.tile(tri_n, (3, 1))
        self.bnd_wgt = np.concatenate(bnd_wgt)
        self.bnd_sum = np.bincount(self.bnd_v, self.bnd_area, n)

        # least-squares gradient weights with omega = 1 / d^2
        src = np.concatenate([a, b])
        dst = np.concatenate([b, a])
        dx = pos[dst] - pos[src]
        omega = 1.0 / np.sum(dx * dx, axis=1)
        mom = np.zeros((n, 3, 3))
        np.add.at(mom, src, omega[:, None, None] * dx[:, :, None] * dx[:, None, :])
        det = np.linalg.det(mom)
        bad = ~(det > 1e-12 * np.trace(mom, axis1=1, axis2=2) ** 3)
        if bad.any():
            raise ValueError(f"Singular least-squares gradient at vertex {int(np.argmax(bad))}")
        self.grad_src = src
        self.grad_dst = dst
        self.grad_w = omega[:, None] * np.einsum("eij,ej->ei", np.linalg.inv(mom)[src], dx)

    # --- vertex and edge helpers -------------------------------------------

    def grad_scl(self, u: np.ndarray) -> np.ndarray:
        """Least-squares gradient (n, 3) of a vertex scalar."""
        du = u[self.grad_dst] - u[self.grad_src]
        n = self.num_tot
        return np.stack([np.bincount(self.grad_src, self.grad_w[:, d] * du, n) for d in range(3)], axis=1)

    def grad_vec(self, v: np.ndarray) -> np.ndarray:
        """Least-squares gradient (n, 3, 3) of a vertex vector; [vid, i, j] = dv_i/dx_j."""
        return np.stack([self.grad_scl(v[:, i]) for i in range(3)], axis=1)

    def face_mean(self, u: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Skew-corrected arithmetic mean on edges."""
        a, b = self.edge[:, 0], self.edge[:, 1]
        return 0.5 * (u[a] + u[b]) + 0.5 * np.sum((g[a] + g[b]) * self.skew, axis=1)

    def face_harm(self, u: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Skew-corrected harmonic mean on edges, guarded against negative values."""
        a, b = self.edge[:, 0], self.edge[:, 1]
        s = u[a] + u[b]
        u_m = np.where(s != 0.0, 2.0 * u[a] * u[b] / np.where(s != 0.0, s, 1.0), 0.0)
        u_f = u_m + 0.5 * np.sum((g[a] + g[b]) * self.skew, axis=1)
        return np.where(u_f > 0.0, u_f, u_m)

    def centroid(self, u: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Vertex value moved to the control volume centroid."""
        return u + np.sum(g * self.cvol_skew, axis=1)

    def net_out(self, q: np.ndarray) -> np.ndarray:
        """Sum of an edge quantity (E,) or (E, k) (positive from a to b) leaving each control volume."""
        if q.ndim == 2:
            return np.stack([self.net_out(q[:, k]) for k in range(q.shape[1])], axis=1)
        n = self.num_tot
        return np.bincount(self.edge[:, 0], q, n) - np.bincount(self.edge[:, 1], q, n)
