import numpy as np
import math
import matplotlib.pyplot as plt

class Median_Data:

    """ Here, assumptions are made that there exists a non-linear mapping which projects the data into a Euclidean Space
    thus, there is a Dissimilarity matrix (interpreted Euclidean Space X_E), which contains the pairwise dissimilarities
    between the data objects. The matrix D is symmetric, where D_ij =  D_ji and D_ii = 0. Also, the prototypes are the
    chosen from the data sample, and new prototypes are selected by the minimisation requirement laid on the cost
    function. """

    def __init__(self, file, number_of_prototype):
        # A M x M Dissimilarity matrix.
        self.D = np.loadtxt(file, delimiter=',')
        self.Distance = np.asarray(self.D)
        # Number of clusters k.
        self.number_of_prototypes = number_of_prototype
        # Length of the Dissimilarity matrix.
        self.lenDist = len(self.Distance)
        self.length = self.lenDist - self.number_of_prototypes
        self.delDistance = self.Distance
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
        # Initialisation of Centering MxM Matrix
        self.CenteringData = np.zeros((self.length, self.length))
        self.sumBetaTyp = np.zeros((self.length, 1))
        # Functions to called initially.
        self.initializeTypicalityMatrix()
        self.initializeMembershipMatrix()
        self.generate_prototype()

    def generate_prototype(self):
        """ The prototypes are updated indirectly using the updated Alpha values.

        :return: The matrix of k x M of actual prototype.
        """
        number_of_rows = self.Distance.shape[0]
        random_indices = np.random.choice(number_of_rows, size=self.number_of_prototypes, replace=False)
        self.prototype = self.Distance[random_indices, :]
        for j in range(self.number_of_prototypes):
            self.deleteDistance(j)

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
            local_error += self.neighborhood_function(j, l) * self.delDistance[i][l]
        return local_error

    def neighborhood_function(self, j, l):
        """
        :param j: contains the index of jth prototype.
        :param l: contains the index of lth prototype.

        @:var rank_lj: int value returned by the ranking function.
        @:var sigma: a double value.
        @:var denominator: a double value.
        @:var numerator: int value containing (rank_lj)^2.
        @:var rank_neighborhood: double value containing the neighborhood value.

        :return: the neighborhood range according to the winning Alpha.
        """
        rank_jl = self.rank_ij(j, l)
        sigma = 1/math.sqrt(2 * math.pi * self.sig * self.sig)
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

        @:returns: the rank of lth prototype according to jth Alpha.
        """
        rank = 0
        for k in range(self.number_of_prototypes):
            if(self.prototype[l][j] - self.prototype[l][k]) > 0:
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
            random_num_list = [np.random.random() for j in range(self.number_of_prototypes)]
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

    def deleteDistance(self, j):
        """
        :param j: Index of new prototype which is needed to be replaced from dissimilarity matrix.
        """
        for i in range(self.length):
            if np.array_equal(self.prototype[j], self.delDistance[i]):
                self.delDistance = np.delete(self.delDistance, i, axis=0)
                break

    def visualise_update(self):
        """ When the coefficients gets updates, this functions visualises where the prototypes are moving.

        :return: An image as output which contains scatter plots.
        """
        plt.scatter(self.Distance[:, 4], self.Distance[:, 2])
        plt.scatter(self.prototype[:, 4], self.prototype[:, 2], marker='x', c='r')
        plt.show()
        plt.close()

    def newPrototype(self, j, l_dash):
        """
        :param j: Index of prototype to be updated.
        :param l_dash: data length
        :return: returns M x 1 vector.
        """
        prodBetaLocalError = 0.0
        typ_sum = 0.0
        for i in range(self.length):
            prodBetaLocalError += self.beta[i][j] * self.local_loss(i, l_dash)
        for i in range(self.length):
            typ_sum += pow((1 - self.typicality_matrix[i][j]), self.typ)
        self.sumBetaTyp[l_dash] = prodBetaLocalError + (self.gamma * typ_sum)
        return self.sumBetaTyp


class MedianNeuralGas:

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
            for j in range(dataset.number_of_prototypes):
                for l_dash in range(dataset.length):
                    arg_min_value = dataset.newPrototype(j, l_dash)
                    dataset.prototype[j] = dataset.Distance[np.argmin(arg_min_value)]
            dataset.delDistance = dataset.Distance
            for j in range(dataset.number_of_prototypes):
                dataset.deleteDistance(j)
            print(dataset.prototype)
            dataset.visualise_update()
            dataset.beta_ij()
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


dataset = Median_Data(r'C:\Users\amitk\Desktop\Programming_Project\D.csv', 3)
neuralgas = MedianNeuralGas(dataset, 8)
print(dataset.beta)