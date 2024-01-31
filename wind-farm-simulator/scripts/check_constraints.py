import numpy as np
from scipy.spatial.distance import pdist


paths = ["wind-farm-simulator/results/envbo_results_WT4.txt",
         "wind-farm-simulator/results/bo_results_WT4.txt",
         "wind-farm-simulator/results/slsqp_results_WT4.txt"]

for path in paths:
    data = np.loadtxt(path, 
                    delimiter=",",
                    skiprows=1)

    for i in range(data.shape[0]):
        xy = data[i, :8].reshape((2, 4)).T
        ds = pdist(xy)
        ds = np.round(ds, 4)
        min_d = min(ds)
        print(f"{min(ds)>=160}: {min_d}")