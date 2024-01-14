import numpy as np

def T(n, N=15):
    # Bell number (N=25 ?)
    total_sum = 0
    for k in range(1, N):
        total_sum += k ** n / np.math.factorial(k)
    return total_sum / np.exp(1)

def define_discrete_dist(u, n):
    # Discrete distribution
    return np.exp(-1) * (u ** n) / np.math.factorial(u)

def sample_discrete_dist(n, u_max=10):
    # Sampling of the number of urns
    density = [define_discrete_dist(u, n) for u in range(u_max)]
    return np.random.choice(a=np.arange(u_max), p=[e / sum(density) for e in density])

def sample_random_partition(n, u_max):
    u = sample_discrete_dist(n, u_max)
    partition = [[] for _ in range(u)]
    for b in range(1, n + 1):
        urn_chosen = np.random.randint(1, u + 1)
        partition[urn_chosen - 1].append(b)

    # Remove empty subsets from the partition
    partition = [subset for subset in partition if subset]

    return partition

for i in range(0,100):
  print(sample_random_partition(3,4))

