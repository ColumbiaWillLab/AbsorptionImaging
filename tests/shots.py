from config import config
from models import shots

import matplotlib.pyplot as plt


path = "/Users/liuhenry/Dev/WillLab/AbsorptionImaging/tests/data/tricky/Raw_20190522-105317"
bmp_paths = [f"{path}_{i}.bmp" for i in range(1, 4)]

shot = shots.Shot(name="path", bmp_paths=bmp_paths)
fit = shot.fit_2D(config)
print(fit.result.fit_report())
fig = plt.figure()
shot.plot(fig)
plt.show()
