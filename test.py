import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from run import main_run, posterior_probability_computation


def generate_random_A(N):
    """Generates a random list of elements in {1,..,N} with random size in {1,...,N}."""
    length_A = np.random.randint(1, N + 1)
    A = []
    while len(A) != length_A:
        a = np.random.randint(1, N + 1)
        if a not in A:
            A.append(a)
    return A


# Random partitions generation with urns method
def T(n, N=15):
    """Returns the n-th Bell or exponential number using the Dobinski's formula truncated with N terms."""
    total_sum = 0
    for k in range(1, N):
        total_sum += k**n / np.math.factorial(k)
    return total_sum / np.exp(1)


def define_discrete_dist(u, n):
    """Returns the probability of having u urns in the urns model, as per (1.4) in https://core.ac.uk/download/pdf/82581547.pdf."""
    return np.exp(-1) * (u**n) / np.math.factorial(u)


def sample_discrete_dist(n, u_max=10):
    """Sample a number of urns in the urns model."""
    density = [define_discrete_dist(u, n) for u in range(u_max)]
    return np.random.choice(a=np.arange(u_max), p=[e / sum(density) for e in density])


def sample_random_partition(n, u_max):
    """Generates a random partition of the set {1,...,n} using at most u_max subsets."""
    u = sample_discrete_dist(n, u_max)
    partition = [[] for _ in range(u)]
    for b in range(1, n + 1):
        urn_chosen = np.random.randint(1, u + 1)
        partition[urn_chosen - 1].append(b)
    # Remove empty subsets from the partition
    partition = [subset for subset in partition if subset]
    return partition


def permutation_inside_sets(partition, seed=123):
    """Permutes randomly the order of the sets elements in a partition."""
    np.random.seed(seed)  # Set the seed.
    perm_partition = []
    for set in partition:
        n = len(set)
        perm_set = permutation_list(set)
        perm_partition.append(perm_set)
    return perm_partition


def permutation_list(list, seed=123):
    """Returns a random permutation of the elements of the list."""
    np.random.seed(seed)  # Set the seed.
    n = len(list)
    permuted_list = []
    indices_permuted = np.random.permutation(n)

    for i in indices_permuted:
        permuted_list.append(list[i])

    return permuted_list


def tests_different_seeds(n_tests, N):
    """Test if the code runs with random arguments."""

    length = []

    for test_number in range(n_tests):

        np.random.seed(test_number)  # Set the seed.
        if N >= 100:
            maxim = 100
        else:
            maxim = N + 1
        partition_1 = sample_random_partition(N, maxim)
        partition_2 = sample_random_partition(N, maxim)
        A = generate_random_A(N)
        w = A[np.random.randint(0, len(A))]
        length.append(main_run(partition_1, partition_2, A, w, visualisations=False))

    length = np.array(length)
    return length


def all_items_equality_check(dict):
    """Check if all items in the dictionary are equal."""
    first = True
    for key in dict.keys():
        if first:
            first_value = dict[key]
            first = False
        else:
            value = dict[key]
            assert value == first_value


def tests_invariance_seeds(n_tests, N):
    """Test if the code runs with random arguments."""
    seed = 501
    seed = seed
    np.random.seed(seed)
    A = generate_random_A(N)
    w = A[np.random.randint(0, len(A))]
    maxim = N + 1
    if N >= 80:
        maxim = 80
    else:
        maxim = N + 1
    partition_1 = sample_random_partition(N, maxim)
    partition_2 = sample_random_partition(N, maxim)

    results_runs = {}

    for test_number in range(n_tests):

        seed = test_number + 1
        perm_partition_1 = permutation_list(partition_1, seed)
        perm_partition_2 = permutation_list(partition_2, seed)

        my_experiment = posterior_probability_computation(
            N=N, partition_1=perm_partition_1, partition_2=perm_partition_2, A=A, w=w
        )
        list_qt_alpha_proba, list_qt_beta_proba, t = my_experiment.run()
        results_runs["{}".format(test_number)] = (
            list_qt_alpha_proba,
            list_qt_beta_proba,
        )

    all_items_equality_check(results_runs)


def tests_invariance_inside_permutation(n_tests, N, ini_seed):
    """Test if the code runs with random arguments."""

    np.random.seed(ini_seed)
    A = generate_random_A(N)
    w = A[np.random.randint(0, len(A))]
    partition_1 = sample_random_partition(N, N + 1)
    partition_2 = sample_random_partition(N, N + 1)

    results_runs = {}

    for test_number in range(n_tests):

        seed = test_number + 1
        perm_partition_1 = permutation_list(partition_1, seed)
        perm_partition_2 = permutation_list(partition_2, seed)

        perm_partition_1 = permutation_inside_sets(perm_partition_1, seed)
        perm_partition_2 = permutation_inside_sets(perm_partition_2, seed)

        my_experiment = posterior_probability_computation(
            N=N, partition_1=perm_partition_1, partition_2=perm_partition_2, A=A, w=w
        )
        list_qt_alpha_proba, list_qt_beta_proba, t = my_experiment.run()
        results_runs["{}".format(test_number)] = (
            list_qt_alpha_proba,
            list_qt_beta_proba,
        )

    all_items_equality_check(results_runs)


if __name__ == "__main__":

    # Extract the arguments using the parser.

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n_tests",
        type=int,
        default=10,
        help="Number of simulations test to run (number of needs).",
    )
    parser.add_argument(
        "-N_min",
        type=int,
        default=10,
        help="Minimum value for N, the size of the universe.",
    )
    parser.add_argument(
        "-N_max",
        type=int,
        default=155,
        help="Maximum value for N, the size of the universe.",
    )
    parser.add_argument(
        "-print_average_length",
        type=bool,
        default=False,
        help="Print the average length (in terms of number of iterations) of the runs.",
    )

    args = parser.parse_args()

    # Run the tests for several universe sizes.

    dic_iteration = {}
    for N in range(
        args.N_min, args.N_max
    ):  # N corresponds to the size of the universe here.
        dic_iteration["{}".format(N)] = 0
        lenght = tests_different_seeds(args.n_tests, N)
        dic_iteration["{}".format(N)] += sum(lenght) / args.n_tests

    if args.print_average_length is True:
        print(
            f"Average length of the runs per value of N (universe size), over {args.n_tests} seeds: {dic_iteration}"
        )

    print("Tests passed successfully.")
