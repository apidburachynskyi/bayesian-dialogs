import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from run import main_run, posterior_probability_computation
from test import generate_random_partition, generate_random_A
import time

def has_crossed(list_qt_alpha_proba, list_qt_beta_proba) :
  """Returns true if there was a crossing, false elsewhere."""
  who_is_larger = []
  for it in range(len((list_qt_alpha_proba))-1):
    if list_qt_alpha_proba[it]>=list_qt_beta_proba[it] :
      who_is_larger.append(0)
    else :
      who_is_larger.append(1)
  for it in range(len(who_is_larger)):
    if who_is_larger[it]!=who_is_larger[0] :
      return True
  return False

def experiment(N_min = 10, N_max = 50, n_tests_per_n = 50) :
  dic_iteration = {}
  for N in range(N_min,N_max+1) : 
    dic_iteration['{}'.format(N)]= 0
    t = 0
    for seed in range(n_tests_per_n) : 
      partition_1 = generate_random_partition(N=N, seed=seed)
      partition_2 = generate_random_partition(N=N, seed=seed+1)
      A = generate_random_A(N)
      w = A[np.random.randint(0,len(A))]
      my_experiment = posterior_probability_computation(N=N,partition_1=partition_1,partition_2=partition_2,A=A,w=w)
      list_qt_alpha_proba, list_qt_beta_proba, lenght,t = my_experiment.run()
      dic_iteration['{}'.format(N)] += t
    dic_iteration['{}'.format(N)]=dic_iteration['{}'.format(N)]/n_tests_per_n
  return dic_iteration

if __name__=='__main__': 
  dic_iteration = experiment(N_min=10,N_max=200,n_tests_per_n=1)
  print(dic_iteration)
  plt.grid()
  plt.plot([key for key in dic_iteration.keys()],[dic_iteration[key] for key in dic_iteration.keys()])
  plt.xticks(np.arange(10, 400, step=50))
  plt.xlabel('Size of the partitions')
  plt.ylabel('Number of crossing (in %)')
  plt.title('Number of crossing as a function of N')
  plt.show()
