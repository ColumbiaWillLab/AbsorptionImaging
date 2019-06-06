import shots
import matplotlib.pyplot as plt

from fitting.utils import gaussian_2D

path = "./tests/data/tricky/Raw_20190522-105317"
bmp_paths = [f"{path}_{i}.bmp" for i in range(1, 4)]

shot = shots.Shot(bmp_paths)
x, y = shot.meshgrid

print(shot.twoD_gaussian.fit_report())
print(shot.twoD_gaussian.var_names)
print(shot.twoD_gaussian.covar[3:5, 3:5])
params = shot.twoD_gaussian.best_values
plt.imshow(shot.absorption, cmap="cividis", vmin=-0.1, vmax=1)
plt.colorbar()
plt.contour(
    shot.twoD_gaussian.eval(x=x, y=y).reshape(shot.shape),
    levels=shot.contour_levels,
    cmap="magma",
    linewidths=1,
    vmin=-0.1,
    vmax=1,
)
plt.axhline(params["y0"], linewidth=0.3)
plt.axvline(params["x0"], linewidth=0.3)
print(shot.contour_levels)
plt.show()

hr, vr = shot.oneD_gaussians
print(vr.fit_report())
vr.plot()

print(hr.fit_report())
hr.plot()
plt.show()

print(shot.atom_number)

# plt.imshow(shot.two_sigma_mask)
# plt.contour(
#     shot.twoD_gaussian.eval(x=x, y=y).reshape(shot.shape),
#     levels=shot.contour_levels,
#     cmap="magma",
#     linewidths=1,
#     vmin=-0.1,
#     vmax=1,
# )
# plt.show()
