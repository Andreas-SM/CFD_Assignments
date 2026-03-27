import numpy as np
import matplotlib.pyplot as plt

# ---------------- VELOCITIES ----------------
def u(x,y):
    return -U0 * np.sin(np.pi * x) * np.cos(np.pi * y)

def v(x,y):
    return U0 * np.cos(np.pi * x) * np.sin(np.pi * y)

# ---------------- BOUNDARY CONDITIONS ----------------
def apply_bc(T,Nx,Ny):
    # West wall: Dirichlet T = T_left
    T[1:Ny+1, 0] = 2.0 * T_left - T[1:Ny+1, 1]

    # East wall: Dirichlet T = T_right
    T[1:Ny+1, Nx+1] = 2.0 * T_right - T[1:Ny+1, Nx]

    # South wall: adiabatic dT/dy = 0
    T[0, :] = T[1, :]

    # North wall: adiabatic dT/dy = 0
    T[Ny+1, :] = T[Ny, :]

# ---------------- PARAMETERS ----------------

def solver(N):
    Nx = Ny = N
    dx = Lx / Nx
    dy = Ly / Ny

    x = np.linspace(0.5 * dx, Lx - 0.5 * dx, Nx)
    y = np.linspace(0.5 * dy, Ly - 0.5 * dy, Ny)
    X, Y = np.meshgrid(x, y)

    # cell-centered velocity, only for quiver plot
    u_cells = -U0 * np.sin(np.pi * X) * np.cos(np.pi * Y)
    v_cells =  U0 * np.cos(np.pi * X) * np.sin(np.pi * Y)

    T = np.full((Ny + 2, Nx + 2), T_right)

    # ---------------- INITIAL CONDITION ----------------
    print("-- Initial conditions")
    T[:, :] = T_right
    apply_bc(T,Nx,Ny)

    # explicit time-step restriction
    # dt_adv = 1.0 / (U0 / dx + U0 / dy)
    # dt_diff = 1.0 / (2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2))
    dt = 1 / ((U0 / dx + U0 / dy) + (2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2)))
    # dt = 0.5 * min(dt_adv, dt_diff)   # safety factor

    tol = 1e-6
    err = 1.0
    t = 0.0
    it = 0
    it_max = 500000

    err_hist = []
    time_hist = []

    # indices nearest to x=0.5 and y=0.5
    i_mid = np.argmin(np.abs(x - 0.5)) + 1   # +1 because of ghost cells
    j_mid = np.argmin(np.abs(y - 0.5)) + 1

    # ---------------- TIME LOOP ----------------

    while err > tol and it < it_max:
        
        T_old = T.copy()

        for j in range(1, Ny + 1):
            yj = (j - 0.5) * dy 
            y_n = j * dy
            y_s = (j - 1) * dy

            for i in range(1, Nx + 1):
                xi = (i - 0.5) * dx
                x_e = i * dx
                x_w = (i - 1) * dx

                # face velocities evaluated analytically
                ue = u(x_e,yj) #-U0 * np.sin(np.pi * x_e) * np.cos(np.pi * yj)
                uw = u(x_w,yj) #-U0 * np.sin(np.pi * x_w) * np.cos(np.pi * yj)
                vn = v(xi,y_n) #U0 * np.cos(np.pi * xi) * np.sin(np.pi * y_n)
                vs = v(xi,y_s) #U0 * np.cos(np.pi * xi) * np.sin(np.pi * y_s)

                Tp = T_old[j, i]
                Te_cell = T_old[j, i + 1]
                Tw_cell = T_old[j, i - 1]
                Tn_cell = T_old[j + 1, i]
                Ts_cell = T_old[j - 1, i]

                # first-order upwind face temperatures
                Te = Tp      if ue > 0.0 else Te_cell
                Tw = Tw_cell if uw > 0.0 else Tp
                Tn = Tp      if vn > 0.0 else Tn_cell
                Ts = Ts_cell if vs > 0.0 else Tp

                conv_x = (ue * Te - uw * Tw) / dx
                conv_y = (vn * Tn - vs * Ts) / dy

                diff_x = alpha * (Te_cell - 2.0 * Tp + Tw_cell) / dx**2
                diff_y = alpha * (Tn_cell - 2.0 * Tp + Ts_cell) / dy**2

                T[j, i] = Tp + dt * (-(conv_x + conv_y) + diff_x + diff_y)

        apply_bc(T,Nx,Ny)

        err = np.max(np.abs(T - T_old))
        t += dt
        it += 1

        err_hist.append(err)
        time_hist.append(t)

        if it % 500 == 0:
            print(f"it = {it:6d}, t = {t:10.5f}, err = {err:.3e}")

    print(f"\nReached steady state at about t = {t:.5f}")
    print(f"Final err = {err:.3e}, iterations = {it}")

    return{
        "T" : T,
        "uc" : u_cells,
        "vc" : v_cells,
        "N": N,
        "x": x,
        "y": y,
        "X" : X,
        "Y" : Y,
        "i_mid": i_mid,
        "j_mid": j_mid,
        "err": err,
        "it": it,
        "t": t,
        "err_hist" : err_hist,
        "time_hist" : time_hist
    }

# ---------------------------------------------
# ---------------- MAIN + PLOT ----------------
# ---------------------------------------------
alpha = 1.0e-3
U0 = 1.0
T_left = 1.0
T_right = 0.0
Lx = Ly = 1.0

N_single=20
results_single={}

N_multiple = [20, 40, 60, 80]

mode = "single"   # or "multiple"

if mode == "single":

    #---------------- SINGLE GRID DIMENSION  ----------------

    results_single=solver(N_single)

    plt.figure()
    plt.imshow(
        results_single['T'][1:-1, 1:-1],
        origin="lower",
        extent=[0.0, Lx, 0.0, Ly],
        cmap="inferno",
        aspect="equal",)
    plt.quiver(results_single['X'], results_single['Y'], results_single['uc'], results_single['vc'], color="white")
    plt.colorbar(label="T")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Steady-state temperature field, t ≈ {results_single['t']:.3f}")
    plt.tight_layout()
    plt.show()

    # ---------------- STATIONARITY HISTORY ----------------
    plt.figure()
    plt.semilogy(results_single['time_hist'], results_single['err_hist'])
    plt.xlabel("t")
    plt.ylabel(r"$\max |T^{n+1}-T^n|$")
    plt.title("Steady-state criterion")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------------- CENTRELINE PROFILES ---------------- 
    plt.figure()
    plt.plot(results_single['x'], (results_single['T'][results_single['j_mid'], 1:-1]+results_single['T'][results_single['j_mid']-1, 1:-1])/2, marker='o')
    plt.xlabel("x")
    plt.ylabel("T")
    plt.title(f"Centreline profile: T(x, y={((results_single['y'][results_single['j_mid']-1]+results_single['y'][results_single['j_mid']-2])/2):.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(results_single['y'], (results_single['T'][1:-1, results_single['i_mid']]+results_single['T'][1:-1, results_single['i_mid']-1])/2, marker='o')
    plt.xlabel("y")
    plt.ylabel("T")
    plt.title(f"Centreline profile: T(x={((results_single['x'][results_single['i_mid']-1]+results_single['x'][results_single['i_mid']-2])/2):.3f}, y)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

elif mode == "multiple":

    #---------------- MULTPLE GRID DIMENSION  ----------------

    results = {}

    for N in N_multiple:
        print(f"Running N = {N}")
        results[N] = solver(N)
        print(f"   iterations = {results[N]['it']}, final err = {results[N]['err']:.3e}")

    # ---------------- PLOT 1: T(x, y=0.5) ----------------
    plt.figure(figsize=(7,5))
    for N in N_multiple:
        plt.plot(results[N]['x'], (results[N]['T'][results[N]['j_mid'], 1:-1]+results[N]['T'][results[N]['j_mid']-1, 1:-1])/2, marker='o', ms=3, label=f"N={N}")
    plt.xlabel("x")
    plt.ylabel("T")
    plt.title("Centreline profile: T(x, y=0.5)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------- PLOT 2: T(x=0.5, y) ----------------
    plt.figure(figsize=(7,5))
    for N in N_multiple:
        plt.plot(results[N]['y'], (results[N]['T'][1:-1, results[N]['i_mid']]+results[N]['T'][1:-1, results[N]['i_mid']-1])/2, marker='o', ms=3, label=f"N={N}") 
    plt.xlabel("y")
    plt.ylabel("T")
    plt.title("Centreline profile: T(x=0.5, y)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()