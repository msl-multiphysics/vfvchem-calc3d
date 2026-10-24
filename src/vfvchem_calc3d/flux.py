"""Discrete face fluxes and control volume terms of the VFVChem operators.

Edge quantities are positive from a to b (see ``Dual3D``); they are antisymmetric,
so internal faces cancel when control volume balances are summed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Dual3D


@dataclass
class Flow:
    """Rhie-Chow flow data of a free fluid; see ``flow_helper::update_flow``."""

    dom: str
    den: np.ndarray  # vid -> density
    pres: np.ndarray  # vid -> pressure
    rc_coeff: np.ndarray  # vid -> V / a_P
    flux: np.ndarray  # edge -> face mass flux from a to b


def flow_flux(
    dual: Dual3D,
    vel: np.ndarray,
    pres: np.ndarray,
    den: np.ndarray,
    visc: np.ndarray,
    fce: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rhie-Chow coefficient V / a_P (n,) and face mass flux (E,)."""
    a, b = dual.edge[:, 0], dual.edge[:, 1]
    face, dist, norm, skew = dual.face, dual.dist, dual.norm, dual.skew
    vol = dual.cvol_vol
    n = dual.num_tot

    # Rhie-Chow coefficient
    # a_P = sum_ngh D_f * A(|Pe_f|) + max(F_f, 0) with the central mass flux F_f
    den_f = dual.face_mean(den, dual.grad_scl(den))
    visc_f = dual.face_harm(visc, dual.grad_scl(visc))
    cond = visc_f * face / dist
    flux_c = den_f * 0.5 * np.sum((vel[a] + vel[b]) * norm, axis=1) * face
    peclet = np.where(cond != 0.0, flux_c / np.where(cond != 0.0, cond, 1.0), 0.0)
    pow_cond = cond * np.maximum(1.0 - 0.1 * np.abs(peclet), 0.0) ** 5
    coeff_p = np.bincount(a, pow_cond + np.maximum(flux_c, 0.0), n) + np.bincount(b, pow_cond + np.maximum(-flux_c, 0.0), n)
    rc = np.where(coeff_p > 0.0, vol / np.where(coeff_p > 0.0, coeff_p, 1.0), 0.0)

    # momentum pressure gradient from the divergence theorem; see Tetrahedron grad_coeff
    # p_f = (p_v + p_n) / 2 + (grad(p_v) + grad(p_n)) / 2 . skew
    pres_f = dual.face_mean(pres, dual.grad_scl(pres))
    pres_sum = dual.net_out((pres_f * face)[:, None] * norm) + pres_bnd(dual, pres)

    # d * (grad(p) - fce) with the body force at the control volume centroid
    mom = rc[:, None] * (pres_sum - vec_source(dual, fce)) / vol[:, None]

    # face mass flux
    # vel_f . n = vel_f(skew) . n - d_f * ((p_n - p_v) / dn - fce_e . n) + (mom_v + mom_n) / 2 . n
    vel_grad = dual.grad_vec(vel)
    vel_f = 0.5 * (vel[a] + vel[b]) + 0.5 * np.einsum("eij,ej->ei", vel_grad[a] + vel_grad[b], skew)
    fce_e = 0.5 * np.sum((fce[a] + fce[b]) * norm, axis=1)
    mom_e = 0.5 * np.sum((mom[a] + mom[b]) * norm, axis=1)
    vel_rc = -0.5 * (rc[a] + rc[b]) * ((pres[b] - pres[a]) / dist - fce_e) + mom_e
    flux = den_f * (np.sum(vel_f * norm, axis=1) + vel_rc) * face
    return rc, flux


def pres_bnd(dual: Dual3D, pres: np.ndarray) -> np.ndarray:
    """Pressure force sum_bnd p_b * n * dA on the boundary sub-faces of each control volume (n, 3).

    p_b = w_v * p_v + w_j * p_j + w_k * p_k at the sub-face centroid
    """
    w = dual.bnd_wgt
    p_b = w[:, 0] * pres[dual.bnd_v] + w[:, 1] * pres[dual.bnd_j] + w[:, 2] * pres[dual.bnd_k]
    out = np.zeros((dual.num_tot, 3))
    np.add.at(out, dual.bnd_v, (p_b * dual.bnd_area)[:, None] * dual.bnd_norm)
    return out


def vec_flux(dual: Dual3D, vel: np.ndarray, pres: np.ndarray, visc: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Conservative face momentum flux (E, 3) from a to b.

    The operators assemble the advection in non-conservative form; the conservative flux differs
    by vel_v * sum_ngh F_f, which is the discrete continuity residual (see ``scl_flux``).
    """
    a, b = dual.edge[:, 0], dual.edge[:, 1]
    face, dist, norm = dual.face, dual.dist, dual.norm

    # advection and diffusion with the power law
    # J_f = F_f * vel_v + (D_f * A(|Pe_f|) + max(-F_f, 0)) * (vel_v - vel_n)
    visc_f = dual.face_harm(visc, dual.grad_scl(visc))
    cond = visc_f * face / dist
    peclet = np.where(cond != 0.0, flux / np.where(cond != 0.0, cond, 1.0), 0.0)
    pow_law = np.maximum(1.0 - 0.1 * np.abs(peclet), 0.0) ** 5
    out = flux[:, None] * vel[a] + (cond * pow_law + np.maximum(-flux, 0.0))[:, None] * (vel[a] - vel[b])

    # transpose and dilatation with the mean of the vertex gradients
    # -mu_f * grad(vel)_f^T . n * dA - lam_f * div(vel)_f * n * dA with lam_f = -2/3 * mu_f
    vel_grad = dual.grad_vec(vel)
    grad_f = 0.5 * (vel_grad[a] + vel_grad[b])
    div_f = grad_f[:, 0, 0] + grad_f[:, 1, 1] + grad_f[:, 2, 2]
    out -= (visc_f * face)[:, None] * np.einsum("eji,ej->ei", grad_f, norm)
    out += (2.0 / 3.0 * visc_f * div_f * face)[:, None] * norm

    # pressure
    pres_f = dual.face_mean(pres, dual.grad_scl(pres))
    out += (pres_f * face)[:, None] * norm
    return out


def vec_source(dual: Dual3D, fce: np.ndarray) -> np.ndarray:
    """Body force integrated over each control volume (n, 3); fce_c * dV with fce_c at the centroid."""
    fce_c = fce + np.einsum("vij,vj->vi", dual.grad_vec(fce), dual.cvol_skew)
    return fce_c * dual.cvol_vol[:, None]


def vec_storage(dual: Dual3D, vel: np.ndarray, vel_prev: np.ndarray, wgt: np.ndarray, dt: float) -> np.ndarray:
    """Backward Euler storage rho_v * (vel_v - vel_v_prev) * dV / dt of each control volume (n, 3)."""
    if not dt > 0.0:
        raise ValueError(f"Time step must be positive, got {dt}")
    return (wgt * dual.cvol_vol / dt)[:, None] * (vel - vel_prev)


def wgt_flux(dual: Dual3D, wgt: np.ndarray, flw: Flow) -> np.ndarray:
    """Weight flux (wgt_f / den_f) * F_f on edges; equals F_f if wgt is the density."""
    wgt_f = dual.face_mean(wgt, dual.grad_scl(wgt))
    den_f = dual.face_mean(flw.den, dual.grad_scl(flw.den))
    return np.where(den_f != 0.0, wgt_f / np.where(den_f != 0.0, den_f, 1.0), 0.0) * flw.flux


def scl_flux(
    dual: Dual3D,
    unk: np.ndarray,
    diffs: list[tuple[np.ndarray, np.ndarray]],
    cond_self: np.ndarray | None,
    flux_w: np.ndarray | None,
) -> np.ndarray:
    """Conservative face flux (E,) of a scalar from a to b.

    diffs - (G_ij vertex values, driving scalar c_j) pairs
    cond_self - G_ii vertex values for the power law; None for pure upwinding
    flux_w - weight flux on edges; None without advection

    The operators assemble sum_ngh a_f * (c_n - c_v) with a_f = D_f * A(|Pe_f|) + max(-W_f, 0);
    the conservative flux J_f = W_f * c_v + a_f * (c_v - c_n) differs from it by c_v * sum_ngh W_f,
    which is the discrete continuity residual.
    """
    a, b = dual.edge[:, 0], dual.edge[:, 1]
    geo = dual.face / dual.dist

    # diffusion
    flux = np.zeros(len(a))
    for diff, drv in diffs:
        flux += dual.face_harm(diff, dual.grad_scl(diff)) * geo * (drv[a] - drv[b])

    # power law correction of the central difference and advection
    if flux_w is not None:
        cond = np.zeros(len(a)) if cond_self is None else dual.face_harm(cond_self, dual.grad_scl(cond_self)) * geo
        peclet = np.where(cond != 0.0, flux_w / np.where(cond != 0.0, cond, 1.0), 0.0)
        pow_law = np.maximum(1.0 - 0.1 * np.abs(peclet), 0.0) ** 5
        flux += flux_w * unk[a] + (cond * (pow_law - 1.0) + np.maximum(-flux_w, 0.0)) * (unk[a] - unk[b])
    return flux


def scl_source(dual: Dual3D, unk: np.ndarray, src: np.ndarray, prd: np.ndarray) -> np.ndarray:
    """Source and production integrated over each control volume (n,).

    R_c * dV + (s_c * c_v + s_v * grad(c_v) . skew) * dV
    """
    src_c = dual.centroid(src, dual.grad_scl(src))
    prd_c = dual.centroid(prd, dual.grad_scl(prd))
    unk_skew = np.sum(dual.grad_scl(unk) * dual.cvol_skew, axis=1)
    return (src_c + prd_c * unk + prd * unk_skew) * dual.cvol_vol


def scl_storage(dual: Dual3D, unk: np.ndarray, unk_prev: np.ndarray, wgt: np.ndarray, dt: float) -> np.ndarray:
    """Backward Euler storage m_v * (c_v - c_v_prev) * dV / dt of each control volume (n,)."""
    if not dt > 0.0:
        raise ValueError(f"Time step must be positive, got {dt}")
    return wgt * (unk - unk_prev) * dual.cvol_vol / dt
