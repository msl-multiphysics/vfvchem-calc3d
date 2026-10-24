__     _________     ______ _                    
\ \   / /  ___\ \   / / ___| |__   ___ _ __ ___  
 \ \ / /| |_   \ \ / / |   | '_ \ / _ \ '_ ` _ \ 
  \ V / |  _|   \ V /| |___| | | |  __/ | | | | |
   \_/  |_|      \_/  \____|_| |_|\___|_| |_| |_|

Vertex-Centered Finite Volume Method Solver for Chemical Engineering Applications
3D Vertex-Centered Finite Volume Method Post-Processing Utility
Copyright (c) 2026 VFVChem Development Team

Overview
--------
This is a post-processing utility for VFVChem, a vertex-centered finite volume method (FVM) solver for chemical engineering applications.
It loads the obj geometry (or bounding box) used by the 3D solver and VFVChem VTU field files, and evaluates domain, boundary, and interface integrals.

Note: The project is under active development. The public API and numerical models may change between releases.

Features
--------
- Loads obj geometries with `dom*`, `bnd*`, and `itf*` groups of planar polygons, or bounding boxes created with `Variables::new_bounds`
- Loads domain VTU PointData (`value`) as NumPy arrays
- Rebuilds the solver's dual mesh (Voronoi control volumes, face areas, skewness, boundary sub-faces, and least-squares gradients) from the VTU tetrahedra
- Interactive geometry visualization with PyVista, including group names, surface triangulations, and surface control volumes
- Control volume averages, minima, and maxima of scalar fields
- Volume integrals of sources and production terms
- Total, advective, and diffusive scalar fluxes on boundaries or interfaces, including cross-diffusion
- Mass flow rates on boundaries or interfaces from the solver's Rhie-Chow face mass fluxes
- Total, advective, pressure, and viscous momentum fluxes (forces of the fluid on walls) on boundaries or interfaces
- Domain balances with outward fluxes per boundary/interface, sources, storage, and closure (scalar, mass, and momentum)
- Steady-state and transient (backward Euler) results

Conservative Fluxes
-------------------
Boundary and interface fluxes are not reconstructed from gradients.
They are obtained from the discrete balance of the control volumes of the boundary nodes:

    flux_out(v) = source(v) - storage(v) - sum_faces J_f(v)

where J_f are the face fluxes of the solver operators (diffusion, power-law advection, and Rhie-Chow mass flux;
for momentum also the transpose and dilatational viscous terms and the pressure).
The fluxes are conservative: they recover Neumann values exactly on the discrete boundary area,
interface fluxes from both sides add to zero, and domain balances close to the solver tolerance.
The advective flux is c_v times the weight flux through the boundary of the control volume; the diffusive flux is the total minus the advective flux.
For momentum, the pressure force uses the solver's sub-face pressure, interpolated at the sub-face centroid of each surface triangle
(w_v * p_v + w_j * p_j + w_k * p_k), and the viscous force is the total minus the advective and pressure parts.

Because the flux of a boundary node is the balance of its control volume, the domain source and storage terms (src, prd, prev) must be given to obtain boundary fluxes.
The properties must be the same as in the solver.

Usage
-----
Geometry and fields:
- `Calc3D(obj_path)` or `Calc3D(x_min, y_min, z_min, x_max, y_max, z_max)` - load the geometry used by the solver
- `show_mesh(show_cvol=False)` - display group names; loaded domains also show their surface triangulation
- `read_scl(dom, path)`, `read_vec(dom, path)` - load scalar or vector VTU fields of a domain (by group name)

Reductions and volume integrals:
- `avg_scl(dom, unk)`, `min_scl(dom, unk)`, `max_scl(dom, unk)`
- `volint_scl_src(dom, src, unk, prd=0.0)` - integral of R + s * c over the control volumes

Mass flow:
- `flow(dom, vel, pres, den, visc, fce=(0.0, 0.0, 0.0))` - Rhie-Chow face mass fluxes of a free fluid
- `surfint_mass(bnd, flw, prev=None)` - outward mass flow rate; prev=(den_prev, dt) for variable density
- `balance_mass(dom, flw, prev=None)`

Scalar transport:
- `surfint_scl_tot(bnd, diff, unk, src=0.0, prd=0.0, wgt=None, flw=None, prev=None)` - total outward flux
- `surfint_scl_adv(bnd, wgt, flw, unk)` - advective outward flux
- `surfint_scl_diff(bnd, diff, unk, src=0.0, prd=0.0, wgt=None, flw=None, prev=None)` - diffusive outward flux
- `balance_scl(dom, diff, unk, src=0.0, prd=0.0, wgt=None, flw=None, prev=None)`

Momentum transport (returns 3-vectors; flw is required):
- `surfint_vec_tot(bnd, visc, vel, fce=(0.0, 0.0, 0.0), flw=None, prev=None)` - total outward momentum flux
- `surfint_vec_adv(bnd, flw, vel)` - advective outward momentum flux
- `surfint_vec_pres(bnd, pres)` - pressure force of the fluid (p n dA)
- `surfint_vec_diff(bnd, visc, vel, fce=(0.0, 0.0, 0.0), flw=None, prev=None)` - viscous force of the fluid (-tau . n dA)
- `balance_vec(dom, visc, vel, fce=(0.0, 0.0, 0.0), flw=None, prev=None)`
The force of the fluid on a wall is `surfint_vec_pres + surfint_vec_diff`.

Arguments:
- Properties (diff, src, prd, wgt) may be floats, per-vertex arrays, or functions of the scalar unknown.
  Sources depending on other fields (e.g., reaction rates) are given as per-vertex arrays.
- `diff` may be a list of (D_ij, c_j) pairs for cross-diffusion.
- `wgt` is the advection weight (e.g., rho * cp); it defaults to the density of `flw`, or 1 without flow.
- `prev=(unk_prev, dt)` adds the storage term with the weight `wgt` (OpSclDomTime).
  `prev=(unk_prev, dt, wgt_time)` sets the storage weight, e.g., the weight of the previous step (OpSclDomTimePrevious) or 1.0 (OpSclDomTimeUnity).
- For momentum, `visc` may be a float, a per-vertex array, or a function of the vertex velocity; `fce` is (3,) or (n, 3).
  `prev=(vel_prev, dt)` uses the density of `flw` as the storage weight (OpVecDomTime).
- Properties of the flow (`den`, `visc` in `flow`) are floats or per-vertex arrays, e.g., computed from a concentration field.
- Interface fluxes are outward from the domain of the field passed.

Requirements
------------
- Python 3.9 or newer
- Dependencies: numpy, meshio, pyvista, matplotlib, and pillow (installed with the package)

Quick Start
-----------
Clone or download the project, then install it in editable mode from the project root:

    pip install -e .

Run an example from the `examples` folder, for instance:

    python examples/calc_heat_threereg.py

This loads the three-region heat example geometry and VTUs and prints boundary, interface, and balance results.

Examples
--------
- `calc_heat_threereg.py` - three regions with interfaces; Neumann recovery and interface fluxes
- `calc_heat_transient.py` - transient heat equation; storage term
- `calc_flow_analytic.py` - plane Poiseuille flow; mass flow rates, wall forces, and momentum balance

Limitations
-----------
- Balances close only to the solver tolerance; the closure measures the residual of the internal control volumes.
  In 3D, the tolerance of the GMRES linear solver usually sets the closure; use `LinearGmres::new_with_options` with a small tolerance for tight balances.
- Transient results require two consecutive time steps (num_ts_write = 1) and the time step size.
- VTU values and coordinates are written with 9 decimals, which limits the accuracy of small fields and of the rebuilt dual mesh.
- Fluxes are summed over the nodes of a group; the discrete boundary area is the surface triangulation through the nodes,
  which cuts the edges and corners of the geometry because the nodes are inset from the face edges.
- The solver assembles advection in non-conservative form; balances include the continuity residual times c_v (or v_v), which vanishes to the solver tolerance.
- For transient flows with variable density, the advective/diffusive (or advective/viscous) split does not include the density storage; the total is unaffected.
- 3D meshes are much larger than 2D meshes; rebuilding the dual mesh and the gradients in Python takes memory and time on fine meshes.
- Porous media flows are not supported.

Project Structure
-----------------
- `src/vfvchem_calc3d/` - `Calc3D` API, geometry and dual mesh, VTU loading, fluxes, and visualization
- `examples/` - example scripts
- `examples/cases/` - sample obj geometries, VFVChem case sources, and VTU outputs

Contributing
------------
Due to limited maintenance capacity, we are not currently accepting external contributions.
However, users are welcome to fork and modify the repository in accordance with the MIT License.

License
-------
VFVChem calc3d is available under the MIT License. See license.txt for details.
