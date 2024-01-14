import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from run import main_run, posterior_probability_computation
from test import generate_random_A, chinese_restaurant_partition

def naive_partitions_from_probs(list_qt_alpha_proba, list_qt_beta_proba, epsilon=0.01, N_max_found=100, N=5, N_max_search=1000,A_arg = None, w_arg = None) :
  """Generates partitions from list posterior probabilities associated to the 
  two agents."""

  list_found = {'A' : [], 'w' : [], 'partition_1' : [], 'partition_2' : []}

  for seed in range(N_max_search) : 
    print(seed)
    if A_arg is None : 
      A = generate_random_A(N)
    else : 
      A = A_arg
    if w_arg is None : 
      w = A[np.random.randint(0,len(A))]
    else : 
      w = w_arg
    partition_1 = chinese_restaurant_partition(N)
    partition_2 = chinese_restaurant_partition(N)
    print("Ok Start")
    print('A: {}'.format(A))
    print('w: {}'.format(w))
    print('partition 1: {}'.format(partition_1))
    print('partition 2: {}'.format(partition_2))
    results_runs = {}
    my_experiment = posterior_probability_computation(N=N,partition_1=partition_1,partition_2=partition_2,A=A,w=w)
    list_qt_alpha_proba_predicted, list_qt_beta_proba_predicted, lenght, t = my_experiment.run()
    if len(list_qt_alpha_proba_predicted)==len(list_qt_alpha_proba) and len(list_qt_beta_proba_predicted)==len(list_qt_beta_proba) : 
      print("Ok first if")
      print('A: {}'.format(A))
      print('w: {}'.format(w))
      print('partition 1: {}'.format(partition_1))
      print('partition 2: {}'.format(partition_2))
      if np.linalg.norm(np.array(list_qt_alpha_proba)-np.array(list_qt_alpha_proba_predicted))<=epsilon and np.linalg.norm(np.array(list_qt_beta_proba)-np.array(list_qt_beta_proba_predicted))<=epsilon :
        print("Ok second if")
        list_found['A'].append(A)
        list_found['w'].append(w)
        list_found['partition_1'].append(partition_1)
        list_found['partition_2'].append(partition_2)
        #print('A: {}'.format(A))
        #print('w: {}'.format(w))
        #print('partition 1: {}'.format(partition_1))
        #print('partition 2: {}'.format(partition_2))
        if len(list_found['A'])>N_max_found : 
          return list_found
  print("Not found")  
  return False

if __name__=='__main__': 
  list_qt_alpha_proba = [1/4, 1/4, 1/4, 1/4, 1/2]
  list_qt_beta_proba = [3/4, 3/4, 3/4, 3/4, 1/2]
  epsilon = 0.01
  N = 20
  N_max_search = 15000
  list_found = naive_partitions_from_probs(list_qt_alpha_proba=list_qt_alpha_proba, 
                              list_qt_beta_proba=list_qt_beta_proba,
                              epsilon=epsilon,
                              N=N,
                              N_max_search=N_max_search,
                              A_arg = None,
                              w_arg = None)
  
  # print("A : {}".format(A))
  # print("w : {}".format(w))
  # print("partition 1 : {}".format(partition_1))
  # print("partition 2 : {}".format(partition_2))

  # my_experiment = posterior_probability_computation(N=N,partition_1=partition_1,partition_2=partition_2,A=A,w=w)
  # list_qt_alpha_proba_predicted, list_qt_beta_proba_predicted = my_experiment.run()
  # print('CHECK')
  # print(list_qt_alpha_proba_predicted)
  # print(list_qt_beta_proba_predicted)
    