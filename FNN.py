import numpy
import scipy.special
from time import time
import matplotlib.pyplot
# %matplotlib inline

class neuralNetwork:

    #init
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #set number of notes per layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #link weight matrices, wih and who
        #weights inside the arrays are w_i_j, where link nodes i to node j in the next layer
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        #learning rate
        self.lr = learningrate

        #activation function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    #train model
    def train(self, inputs_list, targets_list):

        #(optinal) variable of time for the training step time print
        #start = time()

        #convert list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        #calculate into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        #calculate emerging hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #calculate emerging final output layer
        final_outputs = self.activation_function(final_inputs)

        #output layer error (target - actual)
        output_errors = targets - final_outputs
        #hidden layer error is output_errors split by weights recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #update the weights for the links between hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        #update the weights for the links between input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors *hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        #(optional) print of time taken per training step
        #print(time()-start)

        pass

    #query neural network
    def query(self, inputs_list):
        #convert to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        #calculate to hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        #calculate output of the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate final inputs
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #calculate singnals emerging to output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

#number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

#learning rate
learning_rate = 0.3

#create instance
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#load MNIST training data
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#training network
epochs = 10

for e in range(epochs):
    #for epochs iterate over all records of the training data set
    for record in training_data_list:
        #split record by ','
        all_values = record.split(',')
        #scale and shift inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        #create target output values
        target = numpy.zeros(output_nodes) + 0.01
        #all_values[0] is target label for record
        target[int(all_values[0])] = 0.99
        n.train(inputs, target)

    pass

pass

#load MNIST test data
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#test network

#scoreboard for performance
scorecard = []

#go through records of the test data set
for record in test_data_list:
    #split record by ','
    all_values = record.split(',')
    #correct anwnser is first value
    correct_label = int(all_values[0])
    #scale and shift inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #query the network
    outputs = n.query(inputs)
    #the index of highest value corresponds to the label
    label = numpy.argmax(outputs)
    #append correct to incorrect list
    if(label == correct_label):
        #networks awnser matches correct awnser add 1 to scorecard
        scorecard.append(1)
    else:
        #networks awnser doesn't match correct awnser add 0 to scorecard
        scorecard.append(0)
        pass

pass

#calculate the performance score, the fraction of correct awnsers
scorecard_array = numpy.asarray(scorecard)
print("performance =", scorecard_array.sum()/scorecard_array.size)
print("quota =", scorecard_array.sum(), "/", scorecard_array.size)