import shots
import matplotlib.pyplot as plt

from fitting.utils import gaussian_2D

path = "./tests/data/tricky/Raw_20190522-105317"
bmp_paths = [f"{path}_{i}.bmp" for i in range(1, 4)]

shot = shots.Shot(bmp_paths)
x, y = shot.meshgrid

print(shot.twoD_gaussian.fit_report())
params = shot.twoD_gaussian.best_values
plt.imshow(shot.absorption, cmap="gray")
plt.colorbar()
plt.contour(
    gaussian_2D(
        x,
        y,
        params["A"],
        params["x0"],
        params["y0"],
        params["sx"],
        params["sy"],
        params["theta"],
        params["z0"],
    ),
    levels=3,
    cmap="gray",
)
plt.axhline(params["y0"])
plt.axvline(params["x0"])

plt.show()

hr, vr = shot.oneD_gaussians
print(vr.fit_report())
vr.plot()

print(hr.fit_report())
hr.plot()
plt.show()

