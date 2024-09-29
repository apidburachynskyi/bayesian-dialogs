import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from run import main_run, posterior_probability_computation

def generate_random_A(N) :
  """ Generates a random list of elements in {1,..,N} with random size in {1,...,N}."""
  length_A = np.random.randint(1,N+1)
  A = []
  while len(A)!=length_A:
    a = np.random.randint(1,N+1)
    if a not in A :
      A.append(a)   
  return A 

def chinese_restaurant_partition(n):
    """ Returns a random partition."""
    partition = []
    for _ in range(n):
        if len(partition) == 0 or np.random.random() < 1 / (len(partition) + 1):
            partition.append([_+1]) 
        else:
            subset_index = np.random.randint(len(partition))
            partition[subset_index].append(_+1)
    np.random.shuffle(partition)
    return partition

def permutation_inside_sets(partition, seed=123) :
  """Permutes randomly the order of the sets elements in a partition."""
  np.random.seed(seed) # Set the seed. 
  perm_partition = []
  for set in partition :
    n = len(set)
    perm_set = permutation_list(set)
    perm_partition.append(perm_set)
  return perm_partition

def permutation_list(list,seed = 123) :
  """Returns a random permutation of the elements of the list."""
  np.random.seed(seed) # Set the seed. 
  n = len(list)
  permuted_list = []
  indices_permuted = np.random.permutation(n)

  for i in indices_permuted :
    permuted_list.append(list[i])

  return permuted_list

def tests_different_seeds(n_tests,N) :
  """ Test if the code runs with random arguments."""

  length = []

  for test_number in range(n_tests) :

    np.random.seed(test_number) # Set the seed. 
    partition_1 = chinese_restaurant_partition(N)
    partition_2 = chinese_restaurant_partition(N)
    A = generate_random_A(N)
    w = A[np.random.randint(0,len(A))]
    length.append(main_run(N, partition_1, partition_2, A, w, visualisations=False))
    #print({'partition_1: {}'.format(partition_1), 'partition_2: {}'.format(partition_2), 
     #      'A: {}'.format(A), 'w: {}'.format(w)})
    
  length = np.array(length)
  print(f" n_tests {n_tests}")
  print(f" min {np.min(length)}")
  print(f" max {np.max(length)}")
  return length
    
def all_items_equality_check(dict) :
  first = True
  for key in dict.keys() :
    if first :
      first_value = dict[key]
      print(first_value)
      first = False
    else : 
      value = dict[key]
      print(value)
      assert value == first_value
    
def tests_invariance_seeds(n_tests,N,seed=501) :
  """ Test if the code runs with random arguments."""

  seed = seed
  np.random.seed(seed)
  A = generate_random_A(N)
  w = A[np.random.randint(0,len(A))]
  partition_1 = chinese_restaurant_partition(N)
  partition_2 = chinese_restaurant_partition(N)

  results_runs = {}

  for test_number in range(n_tests) :

    seed = test_number+1
    perm_partition_1 = permutation_list(partition_1,seed)
    perm_partition_2 = permutation_list(partition_2,seed)

    my_experiment = posterior_probability_computation(N=N,partition_1=perm_partition_1,partition_2=perm_partition_2,A=A,w=w)
    list_qt_alpha_proba, list_qt_beta_proba = my_experiment.run()
    results_runs['{}'.format(test_number)] = (list_qt_alpha_proba, list_qt_beta_proba)

  all_items_equality_check(results_runs)

  print({'partition_1: {}'.format(partition_1), 'partition_2: {}'.format(partition_2), 
          'A: {}'.format(A), 'w: {}'.format(w)})
  
def tests_invariance_inside_permutation(n_tests,N,ini_seed) :
  """ Test if the code runs with random arguments."""

  np.random.seed(ini_seed)
  A = generate_random_A(N)
  w = A[np.random.randint(0,len(A))]
  partition_1 = chinese_restaurant_partition(N)
  partition_2 = chinese_restaurant_partition(N)

  results_runs = {}

  for test_number in range(n_tests) :

    seed = test_number+1
    perm_partition_1 = permutation_list(partition_1,seed)
    perm_partition_2 = permutation_list(partition_2,seed)

    perm_partition_1 = permutation_inside_sets(perm_partition_1, seed)
    perm_partition_2 = permutation_inside_sets(perm_partition_2, seed)

    my_experiment = posterior_probability_computation(N=N,partition_1=perm_partition_1,partition_2=perm_partition_2,A=A,w=w)
    list_qt_alpha_proba, list_qt_beta_proba = my_experiment.run()
    results_runs['{}'.format(test_number)] = (list_qt_alpha_proba, list_qt_beta_proba)

  all_items_equality_check(results_runs)

  print({'partition_1: {}'.format(partition_1), 'partition_2: {}'.format(partition_2), 
          'A: {}'.format(A), 'w: {}'.format(w)})

if __name__=='__main__': 
  n_tests = 10000
  N = 100
  # tests_invariance_inside_permutation(n_tests,N,ini_seed=1334)
  lenght = tests_different_seeds(n_tests,N)
  print(sum(lenght)/10000)