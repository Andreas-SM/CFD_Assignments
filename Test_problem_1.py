import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def apply_bc(c):
    # West and east boundaries
    for j in range(1, Ny + 1):
        yj = (j - 0.5) * dy

        # inlet: Dirichlet c = 1 on x = 0
        if ymin_in <= yj <= ymax_in:
            c[j, 0] = 2.0 - c[j, 1]
        else:
            c[j, 0] = c[j, 1]

        # outlet: Dirichlet c = 0 on x = 1
        if ymin_out <= yj <= ymax_out:
            c[j, Nx + 1] = -c[j, Nx]
        else:
            c[j, Nx + 1] = c[j, Nx]

    # South / north: homogeneous Neumann
    c[0, :] = c[1, :]
    c[Ny + 1, :] = c[Ny, :]

def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b)

# ---------- MAZE DEFINTION ----------
Lx = Ly = 1.0
Nx = Ny = 50
dx = Lx / Nx
dy = Ly / Ny
x = np.linspace(0.5 * dx, Lx - 0.5 * dx, Nx)
y = np.linspace(0.5 * dy, Ly - 0.5 * dy, Ny)
X, Y = np.meshgrid(x, y) # Cell-center coordinates associated with I(x, y)

script_dir = Path(__file__).resolve().parent
I = np.load(script_dir / "maze_geometry.npy")

im = plt.imshow(
    I,
    origin="lower",
    extent=[0.0, Lx, 0.0, Ly],
    cmap="binary",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="equal",
)

plt.colorbar(im, ticks=[0, 1], label="$I(x,y)$")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

alpha_liq = 1.0e-1
alpha_wall = 1.0e-8

alpha_cc = np.where(I == 0, alpha_liq, alpha_wall)

# ---------- FORWARD EULERO ----------
print('-- Imposing forward eulero conditinos')
alpha_max = alpha_liq
dt_max = (1.0 / (2.0 * alpha_max * (1.0 / dx**2 + 1.0 / dy**2)))*0.95 # safety factor of 0.95
# dt = 0.8 * dt_max   # safety factor

t_end = 50.0
# nt = int(np.ceil(t_end / dt))
nt = int(np.ceil(t_end / dt_max))

# notable time or space coordinates
save_times = [1.0, 5.0, 15.0, 50.0]
saved_fields = {}

c = np.zeros((Ny + 2, Nx + 2)) # define the concentration grid (the points where we evaluate the concentration)

# define alpha
print('-- Defining alpha')
alpha = np.zeros((Ny + 2, Nx + 2))
alpha[1:Ny+1, 1:Nx+1] = alpha_cc # only in the inner grid, no gost cells with alpha

ymin_in, ymax_in = 0.86, 0.94
ymin_out, ymax_out = 0.06, 0.14

# initial conditions and boundary conditions
print('-- Initial conditions')
c[:, :] = 0.0
apply_bc(c)

# ---------- TIME INTEGRATION ----------
plt.figure()

t = 0.0
for n in range(nt):
    apply_bc(c)

    c_old = c.copy()

    for j in range(1, Ny + 1):
        for i in range(1, Nx + 1):
            aP = alpha[j, i]
            aE = alpha[j, i + 1] if i < Nx else aP
            aW = alpha[j, i - 1] if i > 1 else aP
            aN = alpha[j + 1, i] if j < Ny else aP
            aS = alpha[j - 1, i] if j > 1 else aP

            alpha_e = harmonic_mean(aP, aE)
            alpha_w = harmonic_mean(aW, aP)
            alpha_n = harmonic_mean(aP, aN)
            alpha_s = harmonic_mean(aS, aP)

            diff_x = (
                alpha_e * (c_old[j, i + 1] - c_old[j, i])
                - alpha_w * (c_old[j, i] - c_old[j, i - 1])
            ) / dx**2

            diff_y = (
                alpha_n * (c_old[j + 1, i] - c_old[j, i])
                - alpha_s * (c_old[j, i] - c_old[j - 1, i])
            ) / dy**2

            c[j, i] = c_old[j, i] + dt_max * (diff_x + diff_y)
    t += dt_max

    # save requested snapshots
    for ts in save_times:
        if ts not in saved_fields and t >= ts:
            saved_fields[ts] = c[1:Ny+1, 1:Nx+1].copy()

    skip = 7500
    ipause = 0.01
    if n % skip == 0:
        print(t)
        plt.clf()
        maze_walls = np.ma.masked_where(I == 0, I)
        plt.imshow(
            #c,
            c[1:Ny+1, 1:Nx+1],
            origin="lower",
            extent=[0.0, Lx, 0.0, Ly],
            cmap="viridis",
            aspect="equal",
        )
        plt.imshow(
            maze_walls,
            origin="lower",
            extent=[0.0, Lx, 0.0, Ly],
            cmap="gray_r",
            alpha=0.15,
            interpolation="nearest",
            aspect="equal",
        )
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Transient concentration")
        plt.pause(ipause)
        plt.show()
plt.show()



#  ---------- PLOT SAVED TIMES-SHOT --------
for ts in save_times:
    if ts in saved_fields:
        plt.figure()
        maze_walls = np.ma.masked_where(I == 0, I)

        im = plt.imshow(
            saved_fields[ts],
            origin="lower",
            extent=[0.0, Lx, 0.0, Ly],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )

        plt.imshow(
            maze_walls,
            origin="lower",
            extent=[0.0, Lx, 0.0, Ly],
            cmap="gray_r",
            alpha=0.15,
            interpolation="nearest",
            aspect="equal",
        )

        plt.colorbar(im)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"c(x,y,t) at t ≈ {ts}")
        plt.tight_layout()
        plt.show()
    