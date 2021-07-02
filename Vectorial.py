import numpy as np
import math
import imageio
import os
import shutil
from numpy import linalg as la
from sklearn import preprocessing
from scipy.spatial.distance import euclidean as distance
from matplotlib import pyplot as plt


class DataSet:
    """
    The class DataSet initialises the parameters and updates them to calculates the
    values which are used to update the prototypes.
    """

    def __init__(self, file, number_of_prototype, use_Columns):
        # self.data a N x M matrix, where N is the number of rows and M is the number of columns.
        self.data = np.loadtxt(file, delimiter=',', usecols=use_Columns)

        # self.number_of_prototype int value which contains the number of protoype(say K) to represent the data.
        self.number_of_prototypes = number_of_prototype

        # self.length contains the length or the int value of the number rows(N) of the data.
        self.length = len(self.data)

        # self.mem is the fuzzifier of the membership values, ranges usually between 1.2 to 2.0
        self.mem = 1.2

        # self.typ is the
        self.typ = 1.5

        # self.sig is a constant value used in calculation of neighborhood.
        self.sig = 0.5

        # self.a and self.b are positive independent scalars not necessary to sum up to 1.
        self.a = 0.8
        self.b = 0.2

        # self.gamma is a constant.
        self.gamma = 1.0

        # self.typicality_matrix t_ij is N x K dimension matrix whose values are in range [0,1].
        self.typicality_matrix = []

        # self.membership_matrix  is N x K dimension matrix whose values are in range [0, 1]
        # and for any i u_ij the values should sum up to 1.
        self.membership_matrix = []

        # self.generate_vectors generates random vectors.
        self.generate_vectors()

        # self.scale scales the data.
        self.scale()

        # self.initializeTypicalityMatri() and self.initializeMembershipMatrix(): initialises membership
        # and typicality matrix
        self.initializeTypicalityMatrix()
        self.initializeMembershipMatrix()

        # self.beta: initialises an N x K matrix with zeros.
        self.beta = np.zeros((self.length, self.number_of_prototypes))

    def random_vectors(self):
        """Generates random vector same as the dimension of the data.

        @:return a point of 1 x M dimension
        """
        point = self.data[np.random.randint(self.length)]
        return point

    def generate_vectors(self):
        """generate_vectors() iterates through number of prototype K, to generate and
        append the vector as an array in self.neurons.

        @:return a K x M dimension matrix
        """
        neurons = []
        for i in range(self.number_of_prototypes):
            neurons.append(self.random_vectors())
        self.neurons = np.asarray(neurons)
        return self.neurons

    def scale(self):
        # The preprocessing.scale() function puts the data on one scale.
        self.data = preprocessing.scale(self.data)

    def print_dataset(self):
        print(self.data)

    def distance_prototype(self, neuron_j, neuron_l):
        """ distance_protoype(): calculates the rank of the lth neuron according to jth neuron.

            @:param neuron_j: contains the index of jth neuron.
            @:param neuron_l: contains the index of lth neuron.

            @:var prototype_l: contains the 1 x M vector with index neuron_j.
            @:var prototype_j: contains the 1 x M vector with index neuron_l.
            @:var rank_j: everytime adds 1, when the d(w_l,w_j) - d(w_l, w_k) > 0 else 0.

            @:returns: the rank of lth neuron according to jth neuron.
        """
        rank_j = 0
        prototype_l = self.neurons[neuron_l]
        prototype_j = self.neurons[neuron_j]
        for k in range(self.number_of_prototypes):
            distance_lj = distance(prototype_l, prototype_j) - distance(prototype_l, self.neurons[k])
            if distance_lj > 0:
                rank_j = rank_j + 1
        return rank_j

    def neighborhood_function(self, neuron_j, neuron_l):
        """
        :param neuron_j: contains the index of jth neuron.
        :param neuron_l: contains the index of lth neuron.

        @:var rank_lj: int value returned by the ranking function.
        @:var sigma: a double value.
        @:var denominator: a double value.
        @:var numerator: int value containing (rank_lj)^2.
        @:var rank_neighborhood: double value containing the neighborhood value.

        :return: the neighborhood range according to the winning neuron.
        """
        rank_lj = self.distance_prototype(neuron_j, neuron_l)
        sigma = 1/math.sqrt(2 * math.pi * self.sig * self.sig)
        denominator = 2 * self.sig * self.sig
        numerator = rank_lj * rank_lj
        rank_neighborhood = sigma * (pow(math.e, -(numerator / denominator)))
        return rank_neighborhood

    def sqEuclideanDistance(self, data_i, neuron_l):
        """
        :param data_i: contains the index of the ith data.
        :param neuron_l: contains the index of the lth neuron.

        @:var data_i: contains the 1 x M vector of dataset with index data_i.
        @:var neuron_l : contains the 1 x M vector of neurons with index neuron_l.
        @:var distance_il: calculates the squared euclidean distance between data_i and neuron_l.

        :return: euclidean distance between the ith data and the lth neuron.
        """
        data_i = data_i
        neuron_l = neuron_l
        distance_il = math.pow(la.norm(self.data[data_i] - self.neurons[neuron_l]), 2)
        return distance_il

    def local_loss(self, data_i, neuron_j):
        """
        :param data_i: contains the index of the ith data.
        :param neuron_j: contains the index of the jth neuron.

        :var local_error: a double value which contains the sum of loss between data at index data_i
        and all the neurons at index neuron_l.

        :return: local loss between the data at index data_i and neuron at index neuron_j
        """
        local_error = 0.0
        for neuron_l in range(self.number_of_prototypes):
            local_error += self.neighborhood_function(neuron_j, neuron_l) * self.sqEuclideanDistance(data_i, neuron_l)
        return local_error

    def initializeMembershipMatrix(self):
        """
        @:var random_num_list: contains the list of size K with random numbers in range [0.0 , 1.0).
        @:var summation: contains the sum of list.
        @:var temp_list: contains the list which sums up to 1.

        :return: a N x K membership matrix.
        """
        for i in range(self.length):
            random_num_list = [np.random.random() for i in range(self.number_of_prototypes)]
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
            random_num_list = [np.random.random() for i in range(self.number_of_prototypes)]
            self.typicality_matrix.append(random_num_list)
        return self.typicality_matrix

    def beta_ij(self):
        """
        @:var second_term: float value which contains value of typicality[i][l] to the power of typicality constant
         and then multiplied with scalar b.
        @:var first term:  float value which contains value of membership[i][l] to the power of membership constant
         and then multiplied with scalar a.

        :return: a N x K matrix.
        """
        for i in range(self.length):
            for l in range(self.number_of_prototypes):
                second_term = self.b * (math.pow(self.typicality_matrix[i][l], self.typ))
                first_term = self.a * (math.pow(self.membership_matrix[i][l], self.mem))
                self.beta[i][l] = first_term + second_term

    def updatePrototype(self, neuron_j):
        """
        :param neuron_j: index of the neuron which has to be updated.

        @:var num: calculates the numerator part of the prototype update equation.
        @var den: calculates the denominator part of the prototype update equation.
        """
        num = 0.0
        den = 0.0
        self.beta_ij()
        for i in range(self.length):
            for l in range(self.number_of_prototypes):
                num += self.beta[i, l] * self.neighborhood_function(neuron_j, l) * self.data[i]
                den += self.beta[i, l] * self.neighborhood_function(neuron_j, l)
        self.neurons[neuron_j] = np.true_divide(num, den)

    def updateMembershipMatrix(self, data_i, neuron_j):
        """
        updateMembershipMatrix(): this function updates the membership of the data at index i
        according to the neuron at index j.

        :param data_i: index of the ith data.
        :param neuron_j: index of the jth neuron.

        @var power_m: float value.
        @var i: contains data at index data_i.
        @var j: contains neuron at index neuron_j.
        @var numerator: calculates the numerator part of updating the membership equation.
        @var denominator: calculates the denominator part of updating the membership equation.
        @:var div: contains the result after division of the numerator with denominator.
        """
        power_m = float(1 / (self.mem - 1))
        i = data_i
        j = neuron_j
        div = 0.0
        numerator = self.local_loss(i, j)
        for l in range(self.number_of_prototypes):
            denominator = self.local_loss(i, l)
            div += pow((numerator / denominator), power_m)
        temp = 1 / div
        self.membership_matrix[i][j] = temp

    def updateTypicalityMatrix(self, data_i, neuron_j):
        """
        :param data_i: index of the ith data.
        :param neuron_j:  index of the jth neuron.
        @var num: calculates the numerator part of updating the typicality equation.
        @var den: calculates the denominator part of updating the typicality equation
        @var div: contains the result after division of the numerator with denominator.
        """
        power_t = float(1 / (self.typ - 1))
        i = data_i
        j = neuron_j
        num = self.b * self.local_loss(i, j)
        den = self.gamma
        div = 1 + math.pow((num/den), power_t)
        temp = 1 / div
        self.typicality_matrix[i][j] = temp

    def get_nearest_n(self, point):
        """
        :param point: contains the index of the data.

        @:var nearest: initialises nearest to the first value of Beta_ij as nearest.
        @:var nearest_neuron: contains the index of the neuron which is nearest.

        :return: neuron which is nearest to the data.
        """
        nearest = self.beta[point][0]
        nearest_neuron = 0
        for n in range(self.number_of_prototypes):
            val = self.beta[point][n]
            if val > nearest:
                nearest = val
                nearest_neuron = n
        return nearest_neuron

    def generate_plot(self, col1, col2, filename):
        """
        :param col1: first column to be used to plot the data.
        :param col2: second column to be used to plot the data.
        :param filename: file name to be used for the generated image.
        """
        count = 0
        plt.ion()
        plt.clf()
        colors = ['blue', 'yellow', 'green', 'purple', 'orange', 'black', 'brown']
        if not os.path.isdir('tmp'):
            os.makedirs('tmp')
        filename = str(filename).rjust(4, '0')
        for row in self.data:
            n = self.get_nearest_n(count)
            plt.plot(row[col1], row[col2], marker='o', markersize=3, color=colors[n % 7])
            count += 1
        for neuron in self.neurons:
            plt.plot(neuron[col1], neuron[col2], marker='x', markersize=5, color='red')
        plt.show()
        plt.pause(0.2)
        filename = 'tmp/' + filename + '.png'
        plt.savefig(filename)

    def create_animation(self, name):
        """
        create_animation: this function used the image generated by the generate_plot
         to create an animation and outputs a .gif file.

        :param name: name to be used to save the .gif file.
        """
        lst = os.listdir('tmp')
        with imageio.get_writer(name, mode='I', duration=0.1) as writer:
            for filename in lst:
                image = imageio.imread('tmp/' + filename)
                writer.append_data(image)
        writer.close()
        shutil.rmtree('tmp')

class VectorialNeuralGas:

    """
    Vectorial Neural Gas Clustering uses prototypes to represent the data by small
    number of vectors which are the points in data space. Each prototype tries to express
    the distribution of similar data points using the similarity measures. This algorithm
    uses membership (probabilistic) and typicality (possibilistic) values which is really helpful
    when the data is overlapping.
    """

    def __init__(self, data, epochs, generate_plot=False, col1=None, col2=None):
        """
        :param data: contains the dataset which is unlabelled.
        :param epochs: number of epochs to update the prototype.
        :param generate_plot: generates the plot.
        :param col1: first column to be used to plot the data.
        :param col2: second column to be used to plot the data.

        @:var self.dec_sigma: factor to which the sigma value should be decreased.
        """
        self.epoch = epochs
        self.dec_sigma = 0.002
        for e in range(self.epoch):
           if generate_plot:
                dataset.generate_plot(col1, col2, e)
           print('Prototype update at epoch ' + str(e+1))
           for j in range(dataset.number_of_prototypes):
                dataset.updatePrototype(j)
           print(dataset.neurons)
           if (dataset.sig - self.dec_sigma) > 0:
               dataset.sig = dataset.sig - self.dec_sigma
           self.updateMemTyp()
        dataset.create_animation('Vectorialneuralgas.gif')

    def updateMemTyp(self):
        """
        updateMemTyp(): updates the membership values and typicality values after the every
        iteration of the prototype update
        """
        for i in range(dataset.length):
            for j in range(dataset.number_of_prototypes):
                dataset.updateMembershipMatrix(i, j)
                dataset.updateTypicalityMatrix(i, j)

dataset = DataSet(r'C:\Users\amitk\Desktop\Programming_Project\iris.data', 3, (0, 1, 2, 3))
neuralgas = VectorialNeuralGas(dataset, 80, True, 2, 3)