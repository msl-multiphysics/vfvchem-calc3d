from vfvchem_calc3d import Calc3D

# load the bounding box used by the solver (examples/cases/heat_transient/heat_transient.rs)
calc = Calc3D(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

# transient results need two consecutive time steps (num_ts_write = 1)
# temp_0000.vtu is the initial condition; temp_0010.vtu is the result after 10 time steps of dt = 0.1 s
dt = 0.1
T_prev = calc.read_scl("dom", "examples/cases/heat_transient/temp_0009.vtu")  # returns np.array with temperature (T) at t = 0.9 s
T = calc.read_scl("dom", "examples/cases/heat_transient/temp_0010.vtu")  # returns np.array with temperature (T) at t = 1.0 s

# total surface integral (-k * grad(T) . n dA) at t = 1.0 s
# wgt is the coefficient of the scalar in the transport equation: wgt * (dT/dt + v . grad(T)) = div(k * grad(T)) + Q
# for heat transfer wgt = rho * cp; for concentrations per unit mass wgt = rho
q_l = calc.surfint_scl_tot("bnd_l", 1.0, T, src=-500.0, wgt=10.0, prev=(T_prev, dt))  # returns float; outward flux 50 W m-2 times the discrete area
q_r = calc.surfint_scl_tot("bnd_r", 1.0, T, src=-500.0, wgt=10.0, prev=(T_prev, dt))  # returns float
print(q_l)
print(q_r)

# heat balance; storage is the rate of change of the heat content of the domain
bal = calc.balance_scl("dom", 1.0, T, src=-500.0, wgt=10.0, prev=(T_prev, dt))
print(bal)
