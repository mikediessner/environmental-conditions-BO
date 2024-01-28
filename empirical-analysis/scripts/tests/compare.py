import pandas as pd
import numpy as np


func = "levy-25"
dir = f"tests/{func}"


def final_mae(filepath, runs):
    maes = []
    for run in range(runs):
        mae = np.loadtxt(f"{filepath}_mae_run{run+1}.csv", skiprows=1, delimiter=",")[-1, 1]
        maes.append(mae)
    
    return np.array(maes).reshape((-1, 1))

bm = final_mae(f"{dir}/benchmark-{func}", 30)
ei = final_mae(f"{dir}/observation-ei-{func}", 30)
logei = final_mae(f"{dir}/observation-logei-{func}", 30)
ucb_4 = final_mae(f"{dir}/ucb-4-{func}", 30)
ucb_8 = final_mae(f"{dir}/ucb-8-{func}", 30)
ucb_16 = final_mae(f"{dir}/ucb-16-{func}", 30)

comparison = pd.DataFrame(np.hstack([bm, ei, logei, ucb_4, ucb_8, ucb_16]),
                          columns=["bm", "ei", "logei", "ucb_4", "ucb_8", "ucb_16"])

comparison.to_csv(f"tests/plots/comparison-mae-{func}.csv", index=False)