import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from run import main_run, posterior_probability_computation
from test import T, generate_random_A, define_discrete_dist, sample_discrete_dist, sample_random_partition

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
        u_max = 0
        if N+1 >= 101:
            u_max = 100
        else :
            u_max = N+1
        dic_direction1['{}'.format(N)]= 0
        dic_direction2['{}'.format(N)]= 0
        print(N)
        for seed in range(n_tests_per_n) : 
                partition_1 = sample_random_partition(N,u_max)
                partition_2 = sample_random_partition(N,u_max)
                A = generate_random_A(N)
                w = A[np.random.randint(0,len(A))]
                my_experiment = posterior_probability_computation(N=N,partition_1=partition_1,partition_2=partition_2,A=A,w=w)
                list_qt_alpha_proba, list_qt_beta_proba, lenght = my_experiment.run()
                if changes(directions(list_qt_alpha_proba)) >= 1 and changes(directions(list_qt_beta_proba)) >= 1 :
                    continue
                elif changes(directions(list_qt_alpha_proba)) == 1 or changes(directions(list_qt_beta_proba)) == 1 :
                    dic_direction1['{}'.format(N)]+=1
                elif changes(directions(list_qt_alpha_proba)) == 2 or changes(directions(list_qt_beta_proba)) == 2 :
                    dic_direction2['{}'.format(N)]+=1
                    print("P1",partition_1,"P2",partition_2,"A",A,"w",w)
                elif changes(directions(list_qt_alpha_proba)) == 3 or changes(directions(list_qt_beta_proba)) == 3 :
                    continue
        dic_direction1['{}'.format(N)]=(dic_direction1['{}'.format(N)]/n_tests_per_n)*100
        dic_direction2['{}'.format(N)]=(dic_direction2['{}'.format(N)]/n_tests_per_n)*100
    return dic_direction1, dic_direction2


if __name__=='__main__': 
  dic_direction1,dic_direction2,t = experiment(N_min=10,N_max=154,n_tests_per_n=1)
  print(dic_direction1)
  print(dic_direction2)
  plt.grid()
  plt.plot([key for key in dic_direction1.keys()],[dic_direction1[key] for key in dic_direction1.keys()])
  plt.plot([key for key in dic_direction2.keys()],[dic_direction2[key] for key in dic_direction2.keys()])
  plt.xlabel('Size of the partitions')
  plt.xticks(np.arange(10, 154, step=10))
  plt.ylabel('Number of direction changes (in %)')
  plt.title('Number of direction changes as a function of N')
  plt.show()

  
