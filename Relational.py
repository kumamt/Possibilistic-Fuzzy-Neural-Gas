import numpy as np
import math
import matplotlib.pyplot as plt


class Relational_Data:
    """ Here, assumptions are made that there exists a non-linear mapping which projects the data into a Euclidean Space
    thus, there is a Dissimilarity matrix (interpreted Euclidean Space X_E), which contains the pairwise dissimilarities
    between the data objects. The matrix D is symmetric, where D_ij =  D_ji and D_ii = 0. Also, unlike Vectorial data
    there is no direct prototype involved but, they are described indirectly by the coefficient Alpha_j so, by
    updating Alpha virtual prototypes are generated. """

    def __init__(self, file, number_of_prototype):
        # A M x M Dissimilarity matrix.
        self.D = np.loadtxt(file, delimiter=',')
        self.Distance = np.asarray(self.D)
        # Number of clusters k.
        self.number_of_prototypes = number_of_prototype
        # Length of the Dissimilarity matrix.
        self.length = len(self.Distance)
        # Degree of Membership values.
        self.mem = 1.2
        # Degree of Typicality values.
        self.typ = 1.5
        # Constant Scalars.
        self.sig = 0.5
        self.a = 0.8
        self.b = 0.2
        self.gamma = 1.0
        # An M x k matrix contains the degree of D_i belonging to cluster k.
        self.beta = np.zeros((self.length, self.number_of_prototypes))
        # Initialisation of Membership and Typicality lists.
        self.typicality_matrix = []
        self.membership_matrix = []
        self.prototype = np.zeros((self.number_of_prototypes, self.length))
        # Initialisation of Centering MxM Matrix
        self.CenteringData = np.zeros((self.length, self.length))
        # Functions to be called initially.
        self.initializeTypicalityMatrix()
        self.initializeMembershipMatrix()
        self.generate_coefficient()
        self.DoubleCentering()

    def generate_coefficient(self):
        """ This function generates the coefficient Alpha which should range (0,1], with the condition that the sum of
        every single row should be equal to 1.

        :return: k x M matrix
        """
        number_of_rows = self.Distance.shape[0]
        random_indices = np.random.choice(number_of_rows, size=self.number_of_prototypes, replace=False)
        self.Alpha = self.Distance[random_indices, :]
        summ = np.sum(self.Alpha, axis=1)
        self.Alpha = np.divide(self.Alpha.T, summ)
        self.Alpha = self.Alpha.T
        return self.Alpha

    def generate_prototype(self):
        """ The prototypes are updated indirectly using the updated Alpha values.

        :return: The matrix of k x M of actual prototype.
        """
        for j in range(self.number_of_prototypes):
            self.prototype[j] = 0
            for i in range(self.length):
                self.prototype[j] += self.Alpha[j][i] * self.Distance[i]

    def distanceDataNeuron(self, i, j):
        """ The distances between the D_i and Alpha_j for Relational data can be calculated using this function.

        :param i: index of the Dissimilarity matrix.
        :param j: index of the coefficient Alpha.
        :return: Distance between the D_i and Alpha_j.
        """
        first_term = 0.0
        normalise_term = 0.5 * np.dot(self.Alpha[j].T, np.dot(self.Distance, self.Alpha[j]))
        for l in range(self.length):
            first_term += self.Alpha[j][l] * self.Distance[i][l]
        distance_ij = first_term - normalise_term
        return distance_ij

    def CalculateCentering(self, data_r, data_s):
        """ Takes the indexes of the data and returns the value at that particular location.

        :param data_r: index of Dissimilarity data.
        :param data_s: index of the Dissimilarity data.
        :return: Value after double centering the data at index data_r and data_s.
        """
        second_term = 0.0
        third_term = 0.0
        fourth_term = 0.0
        first_term = self.Distance[data_r][data_s]
        for r in range(self.length):
            second_term += self.Distance[r][data_s]
        for s in range(self.length):
            third_term += self.Distance[data_r][s]
        for r in range(self.length):
            for s in range(self.length):
                fourth_term += self.Distance[r][s]
        self.CenteringData[data_r][data_s] = -0.5 * (
                    first_term - (1 / self.length) * second_term - (1 / self.length) * third_term + (
                        1 / (self.length * self.length) * fourth_term))

    def DoubleCentering(self):
        """ Double centering the data means any row or any column should sum to 0. """
        for vr in range(self.length):
            for vs in range(self.length):
                self.CalculateCentering(vr, vs)

    def distance_prototype(self, j, k):
        """ Calculates the distance between the Alphas.

        :param j: jth index of the Alpha.
        :param k: kth index of the Alpha.
        :return: Distance between the Alpha at j and Alpha at k.
        """
        distance_proJK = 0.0
        for vr in range(self.length):
            for vs in range(self.length):
                centering_value = self.CenteringData[vr][vs]
                first_term = self.Alpha[j][vr] * self.Alpha[j][vs]
                second_term = 2 * self.Alpha[j][vr] * self.Alpha[k][vs]
                third_term = self.Alpha[k][vr] * self.Alpha[k][vs]
                pro_value = (first_term + third_term) - second_term
                distance_proJK += centering_value * pro_value
        return distance_proJK

    def local_loss(self, i, j):
        """
        @:var second_term: float value which contains value of typicality[i][l] to the power of typicality constant
              and then multiplied with scalar b.
        @:var first term:  float value which contains value of membership[i][l] to the power of membership constant
              and then multiplied with scalar a.

        :return: a M x k matrix.
        """
        local_error = 0.0
        for l in range(self.number_of_prototypes):
            local_error += self.neighborhood_function(j, l) * self.distanceDataNeuron(i, l)
        return local_error

    def neighborhood_function(self, j, l):
        """
        :param j: contains the index of jth Alpha.
        :param l: contains the index of lth Alpha.

        @:var rank_lj: int value returned by the ranking function.
        @:var sigma: a double value.
        @:var denominator: a double value.
        @:var numerator: int value containing (rank_lj)^2.
        @:var rank_neighborhood: double value containing the neighborhood value.

        :return: the neighborhood range according to the winning Alpha.
        """
        rank_jl = self.rank_ij(j, l)
        sigma = 1 / math.sqrt(2 * math.pi * self.sig * self.sig)
        denominator = 2 * self.sig * self.sig
        numerator = rank_jl * rank_jl
        rank_neighborhood = sigma * (math.pow(math.e, -(numerator / denominator)))
        return rank_neighborhood

    def rank_ij(self, j, l):
        """
        calculates the rank of the lth Alpha according to jth Alpha.

        @:param j: contains the index of jth Alpha.
        @:param l: contains the index of lth Alpha.

        @:var prototype_l: contains the 1 x M vector with index j.
        @:var prototype_j: contains the 1 x M vector with index l.
        @:var rank: everytime adds 1, when the d(l, j) - d(l, k) > 0 else 0.

        @:returns: the rank of lth Alpha according to jth Alpha.
        """
        rank = 0
        distance_lj = self.distance_prototype(l, j)
        for k in range(self.number_of_prototypes):
            if (distance_lj - self.distance_prototype(l, k)) > 0:
                rank = rank + 1
        return rank

    def beta_ij(self):
        """
        @:var second_term: float value which contains value of typicality[i][j] to the power of typicality constant
         and then multiplied with scalar b.
        @:var first term:  float value which contains value of membership[i][j] to the power of membership constant
         and then multiplied with scalar a.

        :return: a M x k matrix.
        """
        for i in range(self.length):
            for j in range(self.number_of_prototypes):
                typ = self.b * (math.pow(self.typicality_matrix[i][j], self.typ))
                mem = self.a * (math.pow(self.membership_matrix[i][j], self.mem))
                self.beta[i][j] = mem + typ

    def initializeMembershipMatrix(self):
        """
        @:var random_num_list: contains the list of size K with random numbers in range [0.0 , 1.0).
        @:var summation: contains the sum of list.
        @:var temp_list: contains the list which sums up to 1.

        :return: a N x K membership matrix.
        """
        for i in range(self.length):
            random_num_list = [np.random.random() for j in range(self.number_of_prototypes)]
            summation = sum(random_num_list)
            temp_list = [num / summation for num in random_num_list]
            self.membership_matrix.append(temp_list)
        return self.membership_matrix

    def initializeTypicalityMatrix(self):
        """
        @:var random_number_list: contains the list of size K with random numbers in range [0.0 , 1.0).
        :return: a N x K typicality matrix.
        """
        for i in range(self.length):
            random_num_list = [np.random.random() for j in range (self.number_of_prototypes)]
            self.typicality_matrix.append(random_num_list)
        return self.typicality_matrix

    def updateMembershipMatrix(self, i, j):
        """
        updates the membership of the data at index i according to the Alpha at index j.

        :param i: index of the ith data.
        :param j: index of the jth Alpha.

        @var power_m: float value.
        @var i: contains data at index i.
        @var j: contains neuron at index j.
        @var numerator: calculates the numerator part of updating the membership equation.
        @var denominator: calculates the denominator part of updating the membership equation.
        @:var div: contains the result after division of the numerator with denominator.
        """
        div = 0.0
        power_mem = float(1 / (self.mem - 1))
        numerator = self.local_loss(i, j)
        for l in range(self.number_of_prototypes):
            denominator = self.local_loss(i, l)
            div += pow(numerator / denominator, power_mem)
        temp = 1 / div
        self.membership_matrix[i][j] = temp

    def updateTypicalityMatrix(self, i, j):
        """
        :param i: index of the ith data.
        :param j:  index of the jth neuron.
        @var num: calculates the numerator part of updating the typicality equation.
        @var den: calculates the denominator part of updating the typicality equation
        @var div: contains the result after division of the numerator with denominator.
        """
        power_typ = 1 / (self.typ - 1)
        num = self.b * self.local_loss(i, j)
        den = self.gamma
        div = 1 + math.pow((num / den), power_typ)
        temp = 1 / div
        self.typicality_matrix[i][j] = temp

    def normaliseAlpha(self):
        """ Everytime the coefficients are updates, there is a need for normalisation of these coefficient
            so that the every row sum should be equal to one
            @:var sum_alpha: takes the sum of row"""
        for p in range(self.number_of_prototypes):
            sum_alpha = 0.0
            for k in range(self.length):
                sum_alpha += self.Alpha[p][k]
            for j in range(self.length):
                self.Alpha[p][j] = self.Alpha[p][j] / sum_alpha

    def visualise_update(self):
        """ When the coefficients gets updates, this functions visualises where the prototypes are moving.

        :return: An image as output which contains scatter plots.
        """
        plt.scatter(self.Distance[:, 2], self.Distance[:, 1])
        plt.scatter(self.prototype[:, 2], self.prototype[:, 1], marker='x', c='r')
        plt.show()
        plt.close()

    def update_coefficient(self, p, q):
        """ This function takes coefficient matrix and updates the values of the coefficient using the Stochastic
        gradient descent learning.

        :param q: index of the qth column of the coefficient matrix.
        :param p: index of the pth row of the coefficient matrix.
        """
        learning_rate = 0.002
        self.beta_ij()
        third_term = 0.0
        final_term = 0.0
        for i in range(self.length):
            beta_ip = self.beta[i][p]
            for l in range(self.number_of_prototypes):
                cal_neighborhood = self.neighborhood_function(p, l)
                third_term += cal_neighborhood * (self.Distance[i] - (self.Distance[q] * self.Alpha[p]))
            final_term += beta_ip * third_term
            third_term = 0.0
        self.Alpha[p] -= learning_rate * final_term

class RelationalNeuralGas:

    def __init__(self, data, epochs):
        """
            :param data: contains the dataset which is unlabelled.
            :param epochs: number of epochs to update the prototype.

            @:var self.dec_sigma: factor to which the sigma value should be decreased.
        """
        self.epoch = epochs
        self.dec_sigma = 0.002
        for e in range(self.epoch):
            print('Prototype update at epoch ' + str(e + 1))
            print(dataset.Alpha)
            for p in range(dataset.number_of_prototypes):
                for q in range(dataset.length):
                    dataset.update_coefficient(p, q)
            dataset.normaliseAlpha()
            dataset.visualise_update()
            dataset.generate_prototype()
            self.updateMembershipTypicality()
            if (dataset.sig - self.dec_sigma) > 0:
                dataset.sig = dataset.sig - self.dec_sigma

    def updateMembershipTypicality(self):
        """
            updateMemTyp(): updates the membership values and typicality values after the every
            iteration of the prototype update.
        """
        for i in range(dataset.length):
            for j in range(dataset.number_of_prototypes):
                dataset.updateMembershipMatrix(i, j)
                dataset.updateTypicalityMatrix(i, j)


dataset = Relational_Data(r'C:\Users\amitk\Desktop\Programming_Project\D.csv', 3)
neuralgas = RelationalNeuralGas(dataset, 7)