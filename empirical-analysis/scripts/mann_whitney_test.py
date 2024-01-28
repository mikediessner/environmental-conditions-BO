from scipy.stats import mannwhitneyu
from source.plot import final_mae


def test(data1, data2):
    stat, p = mannwhitneyu(data1, data2)
    return f"Statistics={stat:.3f}, p={p:.3f}"
bm = final_mae("data/levy/benchmark/benchmark-levy", 30).reshape((-1,))
ei = final_mae("data/levy/ei/ei-levy", 30).reshape((-1,))
logei = final_mae("data/levy/logei/logei-levy", 30).reshape((-1,))
ucb = final_mae("data/levy/ucb-8/ucb-8-levy", 30).reshape((-1,))

print("LEVY")
print("====")
print("EI: ", test(bm, ei))
print("LogEI: ", test(bm, logei))
print("UCB: ", test(bm, ucb))


bm = final_mae("data/hartmann/benchmark/benchmark-hartmann", 30).reshape((-1,))
ei = final_mae("data/hartmann/ei/ei-hartmann", 30).reshape((-1,))
logei = final_mae("data/hartmann/logei/logei-hartmann", 30).reshape((-1,))
ucb = final_mae("data/hartmann/ucb-8/ucb-8-hartmann", 30).reshape((-1,))

print("HARTMANN")
print("========")
print("EI: ", test(bm, ei))
print("LogEI: ", test(bm, logei))
print("UCB: ", test(bm, ucb))
