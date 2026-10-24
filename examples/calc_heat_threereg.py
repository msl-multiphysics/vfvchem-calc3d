from vfvchem_calc3d import Calc3D

# load the obj geometry used by the solver (examples/cases/heat_threereg/heat_threereg.rs)
calc = Calc3D("examples/cases/heat_threereg/obj_threereg.obj")

# load VTU files into domains
T_lo = calc.read_scl("dom_lo", "examples/cases/heat_threereg/temp_lo_0000.vtu")  # returns np.array with temperature (T)
T_m = calc.read_scl("dom_m", "examples/cases/heat_threereg/temp_m_0000.vtu")  # returns np.array with temperature (T)
T_up = calc.read_scl("dom_up", "examples/cases/heat_threereg/temp_up_0000.vtu")  # returns np.array with temperature (T)

# option to view the geometry; loaded domains show their surface triangulation
# calc.show_mesh()

# average, minimum, maximum of a scalar field over a domain
print(calc.avg_scl("dom_m", T_m))
print(calc.min_scl("dom_m", T_m))
print(calc.max_scl("dom_m", T_m))

# volume integral (Q dV)
H_lo = calc.volint_scl_src("dom_lo", -500.0, T_lo)  # returns float
print(H_lo)

# total surface integral (-k * grad(T) . n dA); n is outward unit normal
# the surface triangulation of the nodes cuts the edges, so the discrete area of bnd_lo is below 0.4375 m2
q_lo = calc.surfint_scl_tot("bnd_lo", 1.0, T_lo, src=-500.0)  # returns float; outward flux 50 W m-2 times the discrete area
q_mb = calc.surfint_scl_tot("bnd_mb", 1.0, T_m, src=-500.0)  # returns float
print(q_lo)
print(q_mb)

# interface fluxes are outward from the domain of the scalar field
q_itf_lo = calc.surfint_scl_tot("itf_lo", 1.0, T_lo, src=-500.0)  # returns float
q_itf_m = calc.surfint_scl_tot("itf_lo", 1.0, T_m, src=-500.0)  # returns float
print(q_itf_lo)
print(q_itf_m)

# heat balance of a domain
bal_m = calc.balance_scl("dom_m", 1.0, T_m, src=-500.0)
print(bal_m)
