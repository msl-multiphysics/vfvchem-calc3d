from vfvchem_calc3d import Calc3D

# load the bounding box used by the solver (examples/cases/flow_analytic/flow_analytic.rs)
calc = Calc3D(0.0, 0.0, 0.0, 0.02, 0.005, 0.005)

# load VTU files into domains
vel = calc.read_vec("dom", "examples/cases/flow_analytic/vel_0000.vtu")  # returns (n, 3) np.array with velocity (v)
pres = calc.read_scl("dom", "examples/cases/flow_analytic/pres_0000.vtu")  # returns np.array with pressure (p)

# face mass fluxes of the solver
# arguments: domain name, velocity, pressure, density, viscosity
flw = calc.flow("dom", vel, pres, 1000.0, 0.001)  # returns Flow

# outward mass flow rate; analytic value is rho * (2 / 3) * u_max * H * W = 1.67e-4 kg s-1
# the surface triangulation of the nodes cuts the edges, so the discrete inlet is slightly smaller than H * W
m_l = calc.surfint_mass("bnd_l", flw)  # returns float; inflow is negative
m_r = calc.surfint_mass("bnd_r", flw)  # returns float
print(m_l)
print(m_r)

# mass balance of a domain
bal = calc.balance_mass("dom", flw)
print(bal)

# pressure and viscous surface integrals (p * n dA and -tau . n dA); force of the fluid on the wall
# analytic viscous force is 4 * mu * u_max / H * L * W = 8.0e-7 N in x per wall
F_b_pres = calc.surfint_vec_pres("bnd_b", pres)  # returns np.array shape (3,)
F_b_visc = calc.surfint_vec_diff("bnd_b", 0.001, vel, flw=flw)  # returns np.array shape (3,)
print(F_b_pres)
print(F_b_visc)

# advective surface integral (rho * (v . n) * v dA)
M_l = calc.surfint_vec_adv("bnd_l", flw, vel)  # returns np.array shape (3,)
print(M_l)

# momentum balance of a domain
bal_vec = calc.balance_vec("dom", 0.001, vel, flw=flw)
print(bal_vec)
