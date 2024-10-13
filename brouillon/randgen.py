import numpy as np

def T(n,N=15) :#Bell number (N=25 ?)
  sum = 0
  for k in range(1,N) :
    sum+=k**n/np.math.factorial(k)
  return sum/np.exp(1)

def define_discrete_dist(u,n) :#Discrete dist
  return np.exp(-1)*(u**n)/np.math.factorial(u)

def sample_discrete_dist(n,u_max=10):#Sampling of the number of urns
  density = [define_discrete_dist(u,n) for u in range(u_max)]
  return np.random.choice(a = np.arange(u_max), p = [e/sum(density) for e in density])

def sample_random_partition(n,u_max):
  u = sample_discrete_dist(n,u_max)
  partition = {}
  for b in range(1,n+1) :
    urn_chosen = np.random.randint(1,u+1)
    if str(urn_chosen) in partition.keys() :
      partition[str(urn_chosen)].append(b)
    else :
      partition[str(urn_chosen)]=[]
      partition[str(urn_chosen)].append(b)
  return partition #To convert to a list of list.

print(sample_random_partition(100,100))

