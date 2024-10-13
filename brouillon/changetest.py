import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from run import main_run, posterior_probability_computation
from test import chinese_restaurant_partition, generate_random_A

def directions(lst) :
    """Returns direction of probabilities."""
    directions = []
    for i in range(0,len(lst)-1):
        if lst[i]-lst[i+1] > 0:
            directions.append(-1)
        elif lst[i]-lst[i+1] < 0:
            directions.append(1)
        else : 
            directions.append(0)
    return directions

def changes(lst):
    """Returns number of changes."""
    changes = 0
    for i in range(0,len(lst)-1):
        if lst[i] != lst[i +1] and lst[i+1] != 0 and lst[i] != 0:
            changes += 1
    return changes

def experiment(N_min = 10, N_max = 50, n_tests_per_n = 50) :
    dic_direction1 = {}
    dic_direction2 = {}
    for N in range(N_min,N_max+1) : 
        dic_direction1['{}'.format(N)]= 0
        dic_direction2['{}'.format(N)]= 0
        print(N)
        for seed in range(n_tests_per_n) : 
                partition_1 = chinese_restaurant_partition(N)
                partition_2 = chinese_restaurant_partition(N)
                A = generate_random_A(N)
                w = A[np.random.randint(0,len(A))]
                my_experiment = posterior_probability_computation(N=N,partition_1=partition_1,partition_2=partition_2,A=A,w=w)
                list_qt_alpha_proba, list_qt_beta_proba, lenght, t = my_experiment.run()
                if changes(directions(list_qt_alpha_proba)) >= 1 and changes(directions(list_qt_beta_proba)) >= 1 :
                    continue
                elif changes(directions(list_qt_alpha_proba)) == 1 or changes(directions(list_qt_beta_proba)) == 1 :
                    dic_direction1['{}'.format(N)]+=1
                elif changes(directions(list_qt_alpha_proba)) == 2 or changes(directions(list_qt_beta_proba)) == 2 :
                    dic_direction2['{}'.format(N)]+=1
                elif changes(directions(list_qt_alpha_proba)) == 3 or changes(directions(list_qt_beta_proba)) == 3 :
                    continue
        dic_direction1['{}'.format(N)]=dic_direction1['{}'.format(N)]/n_tests_per_n
        dic_direction2['{}'.format(N)]=dic_direction2['{}'.format(N)]/n_tests_per_n
    return dic_direction1, dic_direction2


if __name__=='__main__': 
  dic_direction1,dic_direction2 = experiment(N_min=50,N_max=400,n_tests_per_n=10000)
  print(dic_direction1)
  print(dic_direction2)
  plt.grid()
  plt.plot([key for key in dic_direction1.keys()],[dic_direction1[key] for key in dic_direction1.keys()])
  plt.plot([key for key in dic_direction2.keys()],[dic_direction2[key] for key in dic_direction2.keys()])
  plt.xticks(np.arange(50, 500, step=50))
  plt.yticks(np.arange(0, 20, step=10))
  plt.xlabel('Size of the partitions')
  plt.ylabel('Number of direction changes (in %)')
  plt.title('Number of direction changes as a function of N')
  plt.show()

  
