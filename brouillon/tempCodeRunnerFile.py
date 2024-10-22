import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml


class posterior_probability_computation:

    def __init__(self, N, partition_1, partition_2, A, w):
        """Initialization for the class.

        Args:
            N (int): size of the universe.
            partition_1 (list): partition of the player alpha.
            partition_2 (list): partition of the player alpha.
            A (list): event we which to know the probability.
            w (int): event that occurs.
        """

        self.N = N
        self.partition_1 = partition_1
        self.partition_2 = partition_2
        self.A = A
        self.w = w

    def P(self, w, partition):
        """Assumption: w is a single a element of the universe. P(w,partition) returns the
        set of partition w belongs to.

        Args:
            w (int): event that occurs.
            partition (list): arbitrary partition.

        Returns:
            set if w belongs to the set, None else.
        """
        for set in partition:  # We select the corresponding sets in the partition
            if w in set:
                return set

        return None

    ### Useful functions

    def intersection(self, set_1, set_2):
        """Returns card(set_1 inter set_2)"""
        list = []
        for i in set_1:
            if i in set_2:
                list.append(i)
        return list

    ### Step 1

    # set = P(w,partition)

    def q1_alpha(self, A, w, partition_1):
        """Return the posterior probability
        P(A | P(w,partition))
        """
        P_alphaw = self.P(w, partition_1)
        return len(self.intersection(A, P_alphaw)) / len(P_alphaw), P_alphaw

    def find_sets0(self, A, w, partition_1, q1):
        """Return the sets of partition such that P(A | set) = proba (q0(A,w,partition))"""
        a_1 = []
        for k in range(len(partition_1)):
            set_k = partition_1[k]
            if len(self.intersection(A, set_k)) / len(set_k) == q1:
                a_1.append(set_k)
        return a_1

    def transforms_sets_of_sets(self, a_1):
        """Turns a set of partition into a single set with all elements.
        E.g., transforms_sets_of_sets([[1,2],[3,4]]) returns [1,2,3,4]"""
        list = []
        for i in range(len(a_1)):
            for e in a_1[i]:
                list.append(e)
        return list

    def q1_beta(self, A, w, partition_2, a_1_transformed):
        """Return thep posterior probability of A for the second individual given its own information
        the communication of the other player probability."""
        P_betaw = self.P(w, partition_2)

        numerator = len(
            self.intersection(self.intersection(A, P_betaw), a_1_transformed)
        )
        denominator = len(self.intersection(P_betaw, a_1_transformed))

        return numerator / denominator, self.intersection(P_betaw, a_1_transformed)

    def find_sets1(self, A, w, a_1_transformed, partition_2, q1):
        """Return the sets of partition such that P(A | set, proba_other_player) = proba (q1(A,w,partition,a_1_transformed))"""
        b_1 = []
        for k in range(len(partition_2)):
            set_k = partition_2[k]
            numerator = len(
                self.intersection(self.intersection(A, set_k), a_1_transformed)
            )
            denominator = len(self.intersection(set_k, a_1_transformed))
            if denominator != 0:
                if numerator / denominator == q1:
                    b_1.append(set_k)
        return b_1

    ### Step t

    def qt_alpha(self, A, w, partition_1, b_prec_transformed):
        """Return the posterior probability
        P(A | P(w,partition), b_prec) at time t.
        """
        P_alphaw = self.P(w, partition_1)
        return len(
            self.intersection(self.intersection(A, P_alphaw), b_prec_transformed)
        ) / len(self.intersection(P_alphaw, b_prec_transformed)), self.intersection(
            P_alphaw, b_prec_transformed
        )

    def find_sets1(self, A, w, a_1_transformed, partition_2, q1):
        """Return the sets of partition such that P(A | set, proba_other_player) = proba (q1(A,w,partition,a_1_transformed))"""
        b_1 = []
        for k in range(len(partition_2)):
            set_k = partition_2[k]
            numerator = len(
                self.intersection(self.intersection(A, set_k), a_1_transformed)
            )
            denominator = len(self.intersection(set_k, a_1_transformed))
            if denominator != 0:
                if numerator / denominator == q1:
                    b_1.append(set_k)
        return b_1

    def a_t(self, A, w, b_prec_transformed, partition_1, qt):
        """Return the sets of partition such that P(A | set, b_t_transformed) = proba (q1(A,w,partition))"""
        a_t = []
        for k in range(len(partition_1)):
            set_k = partition_1[k]
            numerator = len(
                self.intersection(self.intersection(A, set_k), b_prec_transformed)
            )
            denominator = len(self.intersection(set_k, b_prec_transformed))
            if denominator != 0:
                if numerator / denominator == qt:
                    a_t.append(set_k)
        return a_t

    def qt_beta(self, A, w, partition_2, a_t_transformed):
        """Return thep posterior probability of A for the second individual given its own information
        the communication of the other player probability."""
        P_betaw = self.P(w, partition_2)
        list = []

        numerator = len(
            self.intersection(self.intersection(A, P_betaw), a_t_transformed)
        )
        denominator = len(self.intersection(P_betaw, a_t_transformed))

        return numerator / denominator, self.intersection(P_betaw, a_t_transformed)

    def b_t(self, A, w, a_t_transformed, partition_2, qt_beta):
        """Return the sets of partition such that P(A | set, proba_other_player) = proba (q1(A,w,partition,a_1_transformed))"""
        b_t = []
        for k in range(len(partition_2)):
            set_k = partition_2[k]
            numerator = len(
                self.intersection(self.intersection(A, set_k), a_t_transformed)
            )
            denominator = len(self.intersection(set_k, a_t_transformed))
            if denominator != 0:
                if numerator / denominator == qt_beta:
                    b_t.append(set_k)
        return b_t

    def intersection(self, set_1, set_2):
        """Returns card(set_1 inter set_2)"""
        list = []
        for i in set_1:
            if i in set_2:
                list.append(i)
        return list

    def joint_partition(self, partition_1, partition_2):
        """Returns the joint partition of two partitions."""

        joint_partition_output = []

        for i in range(len(partition_1)):
            for j in range(len(partition_2)):
                if len(self.intersection(partition_1[i], partition_2[j])) > 0:
                    joint_partition_output.append(
                        self.intersection(partition_1[i], partition_2[j])
                    )

        return joint_partition_output

    def run(self):

        list_qt_alpha_proba = []
        list_qt_beta_proba = []

        self.join_initial_partition = self.joint_partition(
            self.partition_1, self.partition_2
        )

        # Run

        q1_alpha_proba, partition_ini_alpha = self.q1_alpha(
            self.A, self.w, self.partition_1
        )
        list_qt_alpha_proba.append(q1_alpha_proba)

        ### Prior for beta computation
        elt_part_2 = self.P(self.w, self.partition_2)
        list_qt_beta_proba.append(
            len(self.intersection(self.A, elt_part_2)) / len(elt_part_2)
        )
        ###

        a_1 = self.find_sets0(self.A, self.w, self.partition_1, q1_alpha_proba)
        a_1_transformed = self.transforms_sets_of_sets(a_1)
        q1_beta_proba, partition_ini_beta = self.q1_beta(
            self.A, self.w, self.partition_2, a_1_transformed
        )
        b_1 = self.find_sets1(
            self.A, self.w, a_1_transformed, self.partition_2, q1_beta_proba
        )
        b_1_transformed = self.transforms_sets_of_sets(b_1)

        # print("partitions ini")
        # print(partition_ini_alpha)
        # print(partition_ini_beta)

        t = 1

        qt_alpha_proba = q1_alpha_proba
        qt_beta_proba = q1_beta_proba
        b_before = b_1_transformed

        # print('qt_alpha_proba : {}\nqt_beta_proba : {}\n'.format(np.round(qt_alpha_proba,2),np.round(qt_beta_proba,2)))

        list_qt_alpha_proba.append(qt_alpha_proba)
        list_qt_beta_proba.append(qt_beta_proba)

        qt_alpha_proba, partition_alpha = self.qt_alpha(
            self.A, self.w, self.partition_1, b_before
        )
        at = self.a_t(self.A, self.w, b_before, self.partition_1, qt_alpha_proba)
        a_t_transformed = self.transforms_sets_of_sets(at)
        qt_beta_proba, partition_beta = self.qt_beta(
            self.A, self.w, self.partition_2, a_t_transformed
        )
        bt = self.b_t(self.A, self.w, a_t_transformed, self.partition_2, qt_beta_proba)
        b_before = self.transforms_sets_of_sets(bt)
        t = 2

        # print("partitions")
        # print(partition_alpha)
        # print(partition_beta)
        while qt_alpha_proba != qt_beta_proba:

            # print('qt_alpha_proba : {}\nqt_beta_proba : {}\n'.format(np.round(qt_alpha_proba,2),np.round(qt_beta_proba,2)))

            list_qt_alpha_proba.append(qt_alpha_proba)
            list_qt_beta_proba.append(qt_beta_proba)

            qt_alpha_proba, partition_alpha = self.qt_alpha(
                self.A, self.w, self.partition_1, b_before
            )
            at = self.a_t(self.A, self.w, b_before, self.partition_1, qt_alpha_proba)
            a_t_transformed = self.transforms_sets_of_sets(at)
            qt_beta_proba, partition_beta = self.qt_beta(
                self.A, self.w, self.partition_2, a_t_transformed
            )
            bt = self.b_t(
                self.A, self.w, a_t_transformed, self.partition_2, qt_beta_proba
            )

            # print("partitions")
            # print(partition_alpha)
            # print(partition_beta)
            t += 1
            # print(t)
            b_before = self.transforms_sets_of_sets(bt)
            # print(b_before)
            # print(a_t_transformed)
            print(at)
            print(bt)
            if t > 10:
                break

        list_qt_alpha_proba.append(qt_alpha_proba)
        list_qt_beta_proba.append(qt_beta_proba)
        print(t)

        # print('End\n')
        # print('qt_alpha_proba : {}\nqt_beta_proba : {}\n'.format(np.round(qt_alpha_proba,2),np.round(qt_beta_proba,2)))

        return (
            list_qt_alpha_proba,
            list_qt_beta_proba,
            len(self.joint_partition([partition_alpha], self.join_initial_partition)),
            t,
        )

    def visualisations(self, list_qt_alpha_proba, list_qt_beta_proba):

        Time_steps = np.arange(0, len(list_qt_alpha_proba))
        plt.figure(figsize=(5, 5))
        plt.grid()
        plt.plot(Time_steps, list_qt_alpha_proba, label="alpha")
        plt.plot(Time_steps, list_qt_beta_proba, label="beta")
        plt.xlabel("Iteration number")
        plt.ylabel("Posterior probabilities")
        plt.xticks(np.arange(0, len(list_qt_alpha_proba), step=1))
        plt.legend()
        plt.show()
        plt.close()


### Main
def main_run(N, partition_1, partition_2, A, w, visualisations=True):

    my_experiment = posterior_probability_computation(
        N=N, partition_1=partition_1, partition_2=partition_2, A=A, w=w
    )
    list_qt_alpha_proba, list_qt_beta_proba, lenght, t = my_experiment.run()

    if visualisations:
        my_experiment.visualisations(list_qt_alpha_proba, list_qt_beta_proba)

    return lenght


if __name__ == "__main__":

    ### With the parser

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-N', type=int, default=9, help='Size of the universe')
    # parser.add_argument('-w', type=int, default=1, help='Weight')
    # partition_1 = [[1,2,3],[4,5,6],[7,8,9]]
    # partition_2 = [[1,2,3,4],[5,6,7,8],[9]]
    # A = [1,5,9]
    # args = parser.parse_args()
    # main(N=args.N,partition_1=partition_1,partition_2=partition_2,A=A,w=args.w)

    ### With yaml file

    # Open the YAML file and load its contents
    with open("config.yaml", "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    main_run(
        N=data["N"],
        partition_1=data["partition_1"],
        partition_2=data["partition_2"],
        A=data["A"],
        w=data["w"],
        visualisations=True,
    )
