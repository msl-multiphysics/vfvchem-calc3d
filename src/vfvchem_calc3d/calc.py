"""Calc3D: obj geometry + VFVChem VTU control volume balances."""

from __future__ import annotations

from typing import Any

import numpy as np

from .fields import eval_prop_scl, eval_prop_vec, read_vtu
from .flux import (
    Flow, flow_flux, pres_bnd, scl_flux, scl_source, scl_storage, vec_flux, vec_source, vec_storage, wgt_flux,
)
from .geometry import Dual3D, bounds_groups, on_polygons, read_obj_groups


class Calc3D:
    """Load a VFVChem geometry and evaluate domain, boundary, and interface integrals on VTU fields.

    Boundary and interface fluxes are obtained from the discrete balance of the control volumes
    of their nodes, using the same face fluxes as the solver; they are conservative.
    """

    def __init__(self, *args: Any) -> None:
        if len(args) == 1 and isinstance(args[0], str):
            self._grp2d, self._grp3d = read_obj_groups(args[0])
        elif len(args) == 6:
            self._grp2d, self._grp3d = bounds_groups(*(float(x) for x in args))
        else:
            raise TypeError("Calc3D(obj_path) or Calc3D(x_min, y_min, z_min, x_max, y_max, z_max)")
        if not self._grp3d:
            raise ValueError("Geometry has no dom* groups")

        # VTU coordinates have 9 decimals; nodes are inset from the face edges
        pts = np.concatenate([poly for polys in self._grp3d.values() for poly in polys])
        self._tol = 1e-6 * float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

        self._dual: dict[str, Dual3D] = {}
        self._dom_nodes: dict[str, dict[str, np.ndarray]] = {}  # dom -> group -> vids
        self._field_domain: dict[int, str] = {}

    def show_mesh(self, show_cvol: bool = False) -> None:
        """Display domain, boundary, and interface names; loaded domains show their surface triangulation."""
        from .viz import render_geometry

        render_geometry(self._grp2d, self._grp3d, self._dual, show_cvol=show_cvol)

    # --- loading ----------------------------------------------------------

    def read_scl(self, dom: str, path: str) -> np.ndarray:
        """Load a scalar VTU PointData ``value`` of domain ``dom``."""
        arr = self._read_vtu(dom, path)
        if arr.ndim != 1:
            raise ValueError(f"Expected scalar VTU for read_scl, got shape {arr.shape}")
        return arr

    def read_vec(self, dom: str, path: str) -> np.ndarray:
        """Load a vector VTU PointData ``value`` of domain ``dom``; returns (n, 3)."""
        arr = self._read_vtu(dom, path)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Expected vector VTU shape (n, 3) for read_vec, got {arr.shape}")
        return arr

    def _read_vtu(self, dom: str, path: str) -> np.ndarray:
        if dom not in self._grp3d:
            raise ValueError(f"Unknown domain {dom!r}; available: {sorted(self._grp3d)}")
        pos, tet, values = read_vtu(path)
        if dom not in self._dual:
            self._load_dom(dom, pos, tet)
        else:
            dual = self._dual[dom]
            if pos.shape != dual.pos.shape or np.max(np.abs(pos - dual.pos)) > self._tol:
                raise ValueError(f"VTU {path!r} does not match the vertices of domain {dom!r}")
        out = np.array(values, dtype=float, copy=True)
        self._field_domain[id(out)] = dom
        return out

    def _load_dom(self, dom: str, pos: np.ndarray, tet: np.ndarray) -> None:
        dual = Dual3D(pos, tet)
        nodes = dual.pos[dual.num_pts :]
        if not on_polygons(nodes, self._grp3d[dom], self._tol).all():
            raise ValueError(f"VTU nodes do not lie on the faces of domain {dom!r}")

        # boundaries and interfaces are node sets on their group faces
        groups = {}
        for name, polys in self._grp2d.items():
            mask = on_polygons(nodes, polys, self._tol)
            if mask.any():
                groups[name] = dual.num_pts + np.flatnonzero(mask)
        self._dual[dom] = dual
        self._dom_nodes[dom] = groups

    # --- argument helpers -------------------------------------------------

    def _resolve_field_domain(self, *fields: Any) -> str:
        found: set[str] = set()
        for field in fields:
            if field is None or np.isscalar(field) or callable(field):
                continue
            if id(field) in self._field_domain:
                found.add(self._field_domain[id(field)])
                continue
            arr = np.asarray(field)
            if arr.ndim >= 1:
                matches = [d for d, dual in self._dual.items() if dual.num_tot == arr.shape[0]]
                if len(matches) == 1:
                    found.add(matches[0])
        if not found:
            raise ValueError("Cannot determine domain for field; pass arrays from read_scl/read_vec")
        if len(found) > 1:
            raise ValueError(f"Field arrays belong to multiple domains: {sorted(found)}")
        return next(iter(found))

    def _check_dom(self, dom: str) -> Dual3D:
        if dom not in self._grp3d:
            raise ValueError(f"Unknown domain {dom!r}; available: {sorted(self._grp3d)}")
        if dom not in self._dual:
            raise ValueError(f"Domain {dom!r} has no loaded fields; call read_scl/read_vec first")
        return self._dual[dom]

    def _check_unk(self, dom: str, unk: Any, kind: str) -> np.ndarray:
        arr = np.asarray(unk, dtype=float)
        n = self._dual[dom].num_tot
        if kind == "scl" and arr.shape != (n,):
            raise ValueError(f"Scalar field must have shape ({n},), got {arr.shape}")
        if kind == "vec" and arr.shape != (n, 3):
            raise ValueError(f"Vector field must have shape ({n}, 3), got {arr.shape}")
        return arr

    def _check_flw(self, dom: str, flw: Flow | None) -> Flow | None:
        if flw is not None and flw.dom != dom:
            raise ValueError(f"Flow data belongs to domain {flw.dom!r}, not {dom!r}")
        return flw

    def _group_nodes(self, dom: str, grp: str) -> np.ndarray:
        if grp not in self._grp2d:
            raise ValueError(f"Unknown boundary/interface {grp!r}; available: {sorted(self._grp2d)}")
        if grp not in self._dom_nodes[dom]:
            raise ValueError(f"Boundary/interface {grp!r} has no nodes in domain {dom!r}")
        return self._dom_nodes[dom][grp]

    # --- scalar reductions ------------------------------------------------

    def avg_scl(self, dom: str, unk: Any) -> float:
        """Control volume average sum(u_c * V) / sum(V) over domain ``dom``; u_c is at the CV centroid."""
        dual = self._check_dom(dom)
        u = self._check_unk(dom, unk, "scl")
        u_c = dual.centroid(u, dual.grad_scl(u))
        return float(np.sum(u_c * dual.cvol_vol) / np.sum(dual.cvol_vol))

    def min_scl(self, dom: str, unk: Any) -> float:
        """Vertex minimum of scalar field over domain ``dom``."""
        self._check_dom(dom)
        return float(np.min(self._check_unk(dom, unk, "scl")))

    def max_scl(self, dom: str, unk: Any) -> float:
        """Vertex maximum of scalar field over domain ``dom``."""
        self._check_dom(dom)
        return float(np.max(self._check_unk(dom, unk, "scl")))

    # --- volume integrals -------------------------------------------------

    def volint_scl_src(self, dom: str, src: Any, unk: Any, prd: Any = 0.0) -> float:
        """Volume integral of the source R(u) and production s(u) * u over domain ``dom``."""
        dual = self._check_dom(dom)
        u = self._check_unk(dom, unk, "scl")
        n = dual.num_tot
        src_v = eval_prop_scl(src, u, n, "src")
        prd_v = eval_prop_scl(prd, u, n, "prd")
        return float(np.sum(scl_source(dual, u, src_v, prd_v)))

    # --- flow -------------------------------------------------------------

    def flow(self, dom: str, vel: Any, pres: Any, den: Any, visc: Any, fce: Any = (0.0, 0.0, 0.0)) -> Flow:
        """Rhie-Chow face mass flux of a free fluid in domain ``dom``.

        den, visc - float or per-vertex array; fce - body force (3,) or per-vertex (n, 3) array
        """
        dual = self._check_dom(dom)
        v = self._check_unk(dom, vel, "vec")
        p = self._check_unk(dom, pres, "scl")
        n = dual.num_tot
        den_v = eval_prop_scl(den, None, n, "den")
        visc_v = eval_prop_scl(visc, None, n, "visc")
        fce_v = eval_prop_vec(fce, n, "fce")
        rc, flux = flow_flux(dual, v, p, den_v, visc_v, fce_v)
        return Flow(dom=dom, den=den_v, pres=p, rc_coeff=rc, flux=flux)

    def _mass_out(self, dom: str, flw: Flow, prev: Any) -> tuple[np.ndarray, np.ndarray]:
        dual = self._dual[dom]
        stor = np.zeros(dual.num_tot)
        if prev is not None:
            den_prev, dt = prev
            den_prev = eval_prop_scl(den_prev, None, dual.num_tot, "den_prev")
            stor = scl_storage(dual, flw.den, den_prev, np.ones(dual.num_tot), float(dt))
        return -dual.net_out(flw.flux) - stor, stor

    def surfint_mass(self, bnd: str, flw: Flow, prev: Any = None) -> float:
        """Outward mass flow rate through ``bnd`` from the continuity balance of its nodes.

        prev - (den_prev, dt) for transient flows with variable density
        """
        dom = flw.dom
        self._check_dom(dom)
        vids = self._group_nodes(dom, bnd)
        mass_out, _ = self._mass_out(dom, flw, prev)
        return float(np.sum(mass_out[vids]))

    def balance_mass(self, dom: str, flw: Flow, prev: Any = None) -> dict[str, Any]:
        """Mass balance of domain ``dom``: outflow per group, storage, and closure."""
        dual = self._check_dom(dom)
        self._check_flw(dom, flw)
        mass_out, stor = self._mass_out(dom, flw, prev)
        return self._balance(dom, mass_out, np.zeros(dual.num_tot), stor)

    # --- scalar fluxes ----------------------------------------------------

    def _scl_terms(
        self, dom: str, diff: Any, unk: Any, src: Any, prd: Any, wgt: Any, flw: Flow | None, prev: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Outward flux through the boundary sub-faces of every control volume, source, and storage."""
        dual = self._dual[dom]
        u = self._check_unk(dom, unk, "scl")
        n = dual.num_tot
        self._check_flw(dom, flw)

        # diffusion; a list holds (G_ij, c_j) pairs for cross-diffusion
        pairs = diff if isinstance(diff, list) else [(diff, unk)]
        diffs = []
        cond_self = None
        for coef, drv in pairs:
            coef_v = eval_prop_scl(coef, u, n, "diff")
            diffs.append((coef_v, self._check_unk(dom, drv, "scl")))
            if drv is unk or (cond_self is None and np.array_equal(np.asarray(drv, dtype=float), u)):
                cond_self = coef_v

        # advection; the weight defaults to the density (mass flux)
        flux_w = None
        if wgt is None:
            wgt = flw.den if flw is not None else 1.0
        wgt_v = eval_prop_scl(wgt, u, n, "wgt")
        if flw is not None:
            flux_w = wgt_flux(dual, wgt_v, flw)
        out = dual.net_out(scl_flux(dual, u, diffs, cond_self, flux_w))

        # source and storage
        src_a = scl_source(dual, u, eval_prop_scl(src, u, n, "src"), eval_prop_scl(prd, u, n, "prd"))
        # storage; the weight defaults to the current advection weight
        stor = np.zeros(n)
        if prev is not None:
            if len(prev) not in (2, 3):
                raise ValueError("prev must be (unk_prev, dt) or (unk_prev, dt, wgt_time)")
            u_prev = self._check_unk(dom, prev[0], "scl")
            wgt_t = wgt_v if len(prev) == 2 else eval_prop_scl(prev[2], u, n, "wgt_time")
            stor = scl_storage(dual, u, u_prev, wgt_t, float(prev[1]))
        return src_a - stor - out, src_a, stor

    def _scl_adv(self, dom: str, wgt: Any, flw: Flow, unk: Any) -> np.ndarray:
        """Advective outflow c_v * W_b of every control volume; W_b closes the weight flux of its faces."""
        dual = self._dual[dom]
        u = self._check_unk(dom, unk, "scl")
        self._check_flw(dom, flw)
        wgt_v = eval_prop_scl(flw.den if wgt is None else wgt, u, dual.num_tot, "wgt")
        return -u * dual.net_out(wgt_flux(dual, wgt_v, flw))

    def surfint_scl_tot(
        self, bnd: str, diff: Any, unk: Any, src: Any = 0.0, prd: Any = 0.0,
        wgt: Any = None, flw: Flow | None = None, prev: Any = None,
    ) -> float:
        """Total (diffusive + advective) outward flux through ``bnd`` from the balance of its nodes.

        diff - diffusivity of unk (float, array, or callable of unk), or a list of (diff, driver) pairs
        src, prd - source R and production coefficient s of the domain
        wgt - advection and storage weight (e.g., rho * cp); defaults to the density of flw, or 1
        flw - flow data from ``flow``; None without advection
        prev - (unk_prev, dt) or (unk_prev, dt, wgt_time) for transient results; wgt_time is the storage weight
            and defaults to wgt (OpSclDomTime); pass the weight of the previous step for OpSclDomTimePrevious
            or 1.0 for OpSclDomTimeUnity
        """
        dom = self._resolve_field_domain(unk)
        vids = self._group_nodes(dom, bnd)
        flux_b, _, _ = self._scl_terms(dom, diff, unk, src, prd, wgt, flw, prev)
        return float(np.sum(flux_b[vids]))

    def surfint_scl_adv(self, bnd: str, wgt: Any, flw: Flow, unk: Any) -> float:
        """Advective outward flux sum(c_v * W_b) through ``bnd``; wgt=None uses the density of flw."""
        dom = self._resolve_field_domain(unk)
        vids = self._group_nodes(dom, bnd)
        return float(np.sum(self._scl_adv(dom, wgt, flw, unk)[vids]))

    def surfint_scl_diff(
        self, bnd: str, diff: Any, unk: Any, src: Any = 0.0, prd: Any = 0.0,
        wgt: Any = None, flw: Flow | None = None, prev: Any = None,
    ) -> float:
        """Diffusive outward flux through ``bnd``: total flux minus advective flux; same arguments as surfint_scl_tot."""
        dom = self._resolve_field_domain(unk)
        vids = self._group_nodes(dom, bnd)
        flux_b, _, _ = self._scl_terms(dom, diff, unk, src, prd, wgt, flw, prev)
        if flw is not None:
            flux_b = flux_b - self._scl_adv(dom, wgt, flw, unk)
        return float(np.sum(flux_b[vids]))

    def balance_scl(
        self, dom: str, diff: Any, unk: Any, src: Any = 0.0, prd: Any = 0.0,
        wgt: Any = None, flw: Flow | None = None, prev: Any = None,
    ) -> dict[str, Any]:
        """Scalar balance of domain ``dom``: outward flux per group, source, storage, and closure.

        closure = boundary - source + storage; it is the sum of the residuals of the internal
        control volumes and vanishes to the solver tolerance.
        """
        self._check_dom(dom)
        if self._resolve_field_domain(unk) != dom:
            raise ValueError(f"Scalar field does not belong to domain {dom!r}")
        flux_b, src_a, stor = self._scl_terms(dom, diff, unk, src, prd, wgt, flw, prev)
        return self._balance(dom, flux_b, src_a, stor)

    # --- momentum fluxes --------------------------------------------------

    def _vec_terms(
        self, dom: str, visc: Any, vel: Any, fce: Any, flw: Flow | None, prev: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Outward momentum flux through the boundary sub-faces of every control volume, source, and storage."""
        dual = self._dual[dom]
        v = self._check_unk(dom, vel, "vec")
        n = dual.num_tot
        if flw is None:
            raise ValueError("Momentum fluxes need flow data; pass flw from flow")
        self._check_flw(dom, flw)

        # advection, diffusion, and pressure; the solver density and pressure are in flw
        visc_v = eval_prop_scl(visc, v, n, "visc")
        out = dual.net_out(vec_flux(dual, v, flw.pres, visc_v, flw.flux))

        # body force and storage; the storage weight defaults to the density
        src_a = vec_source(dual, eval_prop_vec(fce, n, "fce"))
        stor = np.zeros((n, 3))
        if prev is not None:
            if len(prev) not in (2, 3):
                raise ValueError("prev must be (vel_prev, dt) or (vel_prev, dt, wgt_time)")
            v_prev = self._check_unk(dom, prev[0], "vec")
            wgt_t = flw.den if len(prev) == 2 else eval_prop_scl(prev[2], None, n, "wgt_time")
            stor = vec_storage(dual, v, v_prev, wgt_t, float(prev[1]))
        return src_a - stor - out, src_a, stor

    def _vec_adv(self, dom: str, flw: Flow | None, vel: Any) -> np.ndarray:
        """Advective momentum outflow vel_v * F_b of every control volume; F_b closes the mass flux of its faces."""
        dual = self._dual[dom]
        v = self._check_unk(dom, vel, "vec")
        if flw is None:
            raise ValueError("Momentum fluxes need flow data; pass flw from flow")
        self._check_flw(dom, flw)
        return -v * dual.net_out(flw.flux)[:, None]

    def surfint_vec_tot(
        self, bnd: str, visc: Any, vel: Any, fce: Any = (0.0, 0.0, 0.0), flw: Flow | None = None, prev: Any = None,
    ) -> np.ndarray:
        """Total (advective + pressure + viscous) outward momentum flux (3,) through ``bnd`` from the balance of its nodes.

        visc - viscosity (float, array, or callable of the vertex velocity)
        fce - body force of the domain; (3,) or (n, 3)
        flw - flow data from ``flow``; required
        prev - (vel_prev, dt) or (vel_prev, dt, wgt_time) for transient results; wgt_time defaults to the density
        """
        dom = self._resolve_field_domain(vel)
        vids = self._group_nodes(dom, bnd)
        flux_b, _, _ = self._vec_terms(dom, visc, vel, fce, flw, prev)
        return np.sum(flux_b[vids], axis=0)

    def surfint_vec_adv(self, bnd: str, flw: Flow, vel: Any) -> np.ndarray:
        """Advective outward momentum flux sum(vel_v * F_b) (3,) through ``bnd``."""
        dom = self._resolve_field_domain(vel)
        vids = self._group_nodes(dom, bnd)
        return np.sum(self._vec_adv(dom, flw, vel)[vids], axis=0)

    def surfint_vec_pres(self, bnd: str, pres: Any) -> np.ndarray:
        """Pressure force sum(p_b * n * dA) (3,) of the fluid on ``bnd``."""
        dom = self._resolve_field_domain(pres)
        vids = self._group_nodes(dom, bnd)
        p = self._check_unk(dom, pres, "scl")
        return np.sum(pres_bnd(self._dual[dom], p)[vids], axis=0)

    def surfint_vec_diff(
        self, bnd: str, visc: Any, vel: Any, fce: Any = (0.0, 0.0, 0.0), flw: Flow | None = None, prev: Any = None,
    ) -> np.ndarray:
        """Viscous force sum(-tau . n * dA) (3,) of the fluid on ``bnd``: total minus advective minus pressure; same arguments as surfint_vec_tot."""
        dom = self._resolve_field_domain(vel)
        vids = self._group_nodes(dom, bnd)
        flux_b, _, _ = self._vec_terms(dom, visc, vel, fce, flw, prev)
        flux_b = flux_b - self._vec_adv(dom, flw, vel) - pres_bnd(self._dual[dom], flw.pres)
        return np.sum(flux_b[vids], axis=0)

    def balance_vec(
        self, dom: str, visc: Any, vel: Any, fce: Any = (0.0, 0.0, 0.0), flw: Flow | None = None, prev: Any = None,
    ) -> dict[str, Any]:
        """Momentum balance of domain ``dom``: outward momentum flux per group, body force, storage, and closure (3-vectors)."""
        self._check_dom(dom)
        if self._resolve_field_domain(vel) != dom:
            raise ValueError(f"Velocity field does not belong to domain {dom!r}")
        flux_b, src_a, stor = self._vec_terms(dom, visc, vel, fce, flw, prev)
        return self._balance(dom, flux_b, src_a, stor)

    def _balance(self, dom: str, flux_b: np.ndarray, src_a: np.ndarray, stor: np.ndarray) -> dict[str, Any]:
        dual = self._dual[dom]
        groups = self._dom_nodes[dom]

        # scalar balances give floats; momentum balances give 3-vectors
        def total(q: np.ndarray) -> Any:
            return np.sum(q, axis=0) if q.ndim == 2 else float(np.sum(q))

        out: dict[str, Any] = {"groups": {name: total(flux_b[vids]) for name, vids in groups.items()}}
        assigned = np.zeros(dual.num_tot, dtype=bool)
        for vids in groups.values():
            assigned[vids] = True
        unassigned = ~assigned
        unassigned[: dual.num_pts] = False
        if unassigned.any():
            out["unassigned"] = total(flux_b[unassigned])
        out["boundary"] = total(flux_b[dual.num_pts :])
        out["source"] = total(src_a)
        out["storage"] = total(stor)
        out["closure"] = out["boundary"] - out["source"] + out["storage"]
        return out
