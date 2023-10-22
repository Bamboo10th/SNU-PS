import numpy as np


def generate_random_numbers(n, m, s=5):
    # 生成均值为n，方差为m的正态分布随机数
    random_numbers = np.random.normal(n, np.sqrt(m), s)
    return random_numbers


l = [0.9858036676470382, 0.2734420689189797, 0.128357902346, 0.15837410607344873, 0.25771614046277175]

l3 = []
m = 0.00006


for i in l:
    random_nums = generate_random_numbers(i, m)
    print(random_nums)
