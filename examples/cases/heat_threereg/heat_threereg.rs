use vfvchem_vfvm3d::*;
use std::fs::create_dir_all;

/// Steady-state heat equation in three regions joined by interfaces.
/// Run with: `cargo run --release --example heat_threereg`
///
/// Geometry:
/// Cube with a lower-front-left and an upper-back-right corner cuboid; see examples/obj/obj_threereg.py.
///
///                  +------+------+
///                 /       /dom_up|
///                +-------+------+|
///               /              / |
///              +--------------+  |
///              |    dom_m     |  +
///              |+------+      | /
///              ||dom_lo|      |/
///              ++------+------+
///
/// Domains:
/// dom_lo - lower-front-left region (0.75 m x 0.25 m x 0.25 m)
/// dom_m - middle region (1.0 m x 1.0 m x 1.0 m cube minus the corner regions)
/// dom_up - upper-back-right region (0.75 m x 0.25 m x 0.25 m)
///
/// Boundaries:
/// bnd_lo - outer faces of dom_lo (left, front, and bottom)
/// bnd_mb - outer faces of dom_m (bottom, front, and right)
/// bnd_mt - outer faces of dom_m (top, back, and left)
/// bnd_up - outer faces of dom_up (right, back, and top)
///
/// Interfaces:
/// itf_lo - between dom_lo and dom_m
/// itf_up - between dom_up and dom_m
///
/// Properties:
/// dom_lo, dom_m, dom_up - thermal conductivity (k = 1.0 W m-1 K-1)
/// dom_lo, dom_m, dom_up - heat production coefficient (s = 0.0 W m-3 K-1)
/// dom_lo, dom_m, dom_up - heat source (Q = -500.0 W m-3)
///
/// Boundary conditions:
/// bnd_lo - outward flux (q = 50.0 W m-2)
/// bnd_mb - temperature (T = 300 K)
/// bnd_mt - no flux (q = 0.0 W m-2)
/// bnd_up - temperature (T = 400 K)
///
/// Interface conditions:
/// itf_lo - temperature and flux continuity
/// itf_up - temperature and flux continuity
///
fn main() -> Result<(), VFVChemError> {
    // output directory
    create_dir_all("examples/output_heat_threereg").unwrap();

    // nodes, points, and variables
    // new_obj - import boundary, interface, and domain surfaces from an obj file
    // arguments: input_file, h_min, h_max, ref_len, max_try, seed
    // h_min - point spacing at the boundary; h_max - point spacing far from the boundary
    // ref_len - distance from the boundary over which the spacing grows from h_min to h_max
    // max_try - attempts to place a point before giving up; seed - random seed for point sampling
    let mut vars = Variables::new_obj("examples/obj/obj_threereg.obj", 0.05, 0.05, 0.05, 100, 42)?;

    // geometry
    // domains, boundaries, and interfaces are selected by their group names in the obj file
    // add_bnd - arguments: domain, group_name
    // add_itf - arguments: domain_a, domain_b, group_name
    let dom_lo = vars.add_dom("dom_lo")?;  // lower-front-left
    let dom_m = vars.add_dom("dom_m")?;  // middle
    let dom_up = vars.add_dom("dom_up")?;  // upper-back-right
    let bnd_lo = vars.add_bnd(dom_lo, "bnd_lo")?;  // left, front, and bottom of dom_lo
    let bnd_mb = vars.add_bnd(dom_m, "bnd_mb")?;  // bottom, front, and right of dom_m
    let bnd_mt = vars.add_bnd(dom_m, "bnd_mt")?;  // top, back, and left of dom_m
    let bnd_up = vars.add_bnd(dom_up, "bnd_up")?;  // right, back, and top of dom_up
    let itf_lo = vars.add_itf(dom_lo, dom_m, "itf_lo")?;  // lower interface
    let itf_up = vars.add_itf(dom_up, dom_m, "itf_up")?;  // upper interface

    // unknown domain values
    // arguments: domain, initial_value, output_file
    // initial_value - initial guess for steady-state problems; initial value for transient problems
    // output_file - can be .csv or .vtu; if empty string, no file is written
    // output index is automatically appended to end of output file name (e.g., temp_lo_0000.vtu)
    let temp_lo = vars.add_scldom_unkn(dom_lo, 0.0, "examples/output_heat_threereg/temp_lo.vtu")?;
    let temp_m = vars.add_scldom_unkn(dom_m, 0.0, "examples/output_heat_threereg/temp_m.vtu")?;
    let temp_up = vars.add_scldom_unkn(dom_up, 0.0, "examples/output_heat_threereg/temp_up.vtu")?;

    // constant domain values
    // arguments: domain, value, output_file
    // heat production is multiplied by the temperature; heat source adds the source directly
    // use heat source by default; use heat production (or both) if heat source does not converge
    let cond_lo = vars.add_scldom_cnst(dom_lo, 1.0, "")?;  // thermal conductivity
    let cond_m = vars.add_scldom_cnst(dom_m, 1.0, "")?;  // thermal conductivity
    let cond_up = vars.add_scldom_cnst(dom_up, 1.0, "")?;  // thermal conductivity
    let hprd_lo = vars.add_scldom_cnst(dom_lo, 0.0, "")?;  // heat production coefficient (s; positive if source; negative if sink)
    let hprd_m = vars.add_scldom_cnst(dom_m, 0.0, "")?;  // heat production coefficient (s; positive if source; negative if sink)
    let hprd_up = vars.add_scldom_cnst(dom_up, 0.0, "")?;  // heat production coefficient (s; positive if source; negative if sink)
    let hsrc_lo = vars.add_scldom_cnst(dom_lo, -500.0, "")?;  // heat source (Q; positive if source; negative if sink)
    let hsrc_m = vars.add_scldom_cnst(dom_m, -500.0, "")?;  // heat source (Q; positive if source; negative if sink)
    let hsrc_up = vars.add_scldom_cnst(dom_up, -500.0, "")?;  // heat source (Q; positive if source; negative if sink)

    // constant boundary values
    // arguments: boundary, value, output_file
    let hflx_lo = vars.add_sclbnd_cnst(bnd_lo, 50.0, "")?;  // heat flux (positive if outward; negative if inward)
    let temp_mb = vars.add_sclbnd_cnst(bnd_mb, 300.0, "")?;  // temperature
    let hflx_mt = vars.add_sclbnd_cnst(bnd_mt, 0.0, "")?;  // heat flux
    let temp_bu = vars.add_sclbnd_cnst(bnd_up, 400.0, "")?;  // temperature

    // steady-state heat transfer solver
    // add_heat_dom - register domain with heat transfer
    // add_hflx_bnd - register boundary with heat flux
    // add_temp_bnd - register boundary with temperature
    // add_hcnt_itf - register interface with temperature and flux continuity
    let mut phys = SteadyHeat::new()?;
    phys.add_heat_dom(dom_lo, temp_lo, cond_lo, hprd_lo, hsrc_lo)?;  // arguments: domain, T, k, s, Q
    phys.add_heat_dom(dom_m, temp_m, cond_m, hprd_m, hsrc_m)?;  // arguments: domain, T, k, s, Q
    phys.add_heat_dom(dom_up, temp_up, cond_up, hprd_up, hsrc_up)?;  // arguments: domain, T, k, s, Q
    phys.add_hflx_bnd(bnd_lo, hflx_lo)?;  // arguments: boundary, q
    phys.add_temp_bnd(bnd_mb, temp_mb)?;  // arguments: boundary, T
    phys.add_hflx_bnd(bnd_mt, hflx_mt)?;  // arguments: boundary, q
    phys.add_temp_bnd(bnd_up, temp_bu)?;  // arguments: boundary, T
    phys.add_hcnt_itf(itf_lo)?;  // arguments: interface
    phys.add_hcnt_itf(itf_up)?;  // arguments: interface

    // physics solver
    // LinearGmres::new - Jacobi-preconditioned restarted GMRES linear solver; arguments: num_thread
    // GMRES is much faster than LU (LinearLu::new) in 3D because the LU factors fill in
    // arguments: vars, linear_solver, max_iter, tol, damping_factor
    // damping_factor - between 0.0 and 1.0; lower for stability and higher for speed (if linear or nearly linear)
    // for highly non-linear problems, using a lower damping factor (e.g., 0.8-0.9) may be faster
    let mut lin = LinearGmres::new(1)?;
    phys.solve(&mut vars, &mut lin, 10, 1e-3, 1.0)?;

    Ok(())
}
