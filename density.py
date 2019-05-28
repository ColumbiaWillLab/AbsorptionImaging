import numpy as np


def atom_number(shot):
    mag = 2.5  # magnification
    pixelsize = 3.75e-3  # 3.75 um, reported in mm.
    lam = 589.158e-9  # resonant wavelength
    delta = 2 * np.pi * 8  # beam detuning in MHz
    Gamma = 2 * np.pi * 9.7946  # D2 linewidth in MHz

    # sodium and camera parameters
    sigma_0 = (3.0 / (2.0 * np.pi)) * (lam) ** 2  # cross-section
    sigma = sigma_0 / (1 + (delta / (Gamma / 2)) ** 2)  # off resonance
    area = (pixelsize * 1e-3 * mag) ** 2  # pixel area in SI units

    density = -np.log(shot.transmission)
    return (area / sigma) * np.sum(density)
