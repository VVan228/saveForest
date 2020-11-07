import random

import numpy
import scipy.special
import json
from PIL import Image


class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, hiddennodes3, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2
        self.hnodes3 = hiddennodes3
        self.onodes = outputnodes

        self.wih1 = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes1, self.inodes))
        self.wh1h2 = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.hnodes2, self.hnodes1))
        self.wh2h3 = numpy.random.normal(0.0, pow(self.hnodes3, -0.5), (self.hnodes3, self.hnodes2))
        self.wh3o = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes3))
        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden1_inputs = numpy.dot(self.wih1, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        hidden3_inputs = numpy.dot(self.wh2h3, hidden2_outputs)
        hidden3_outputs = self.activation_function(hidden3_inputs)

        final_inputs = numpy.dot(self.wh3o, hidden3_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden3_errors = numpy.dot(self.wh3o.T, output_errors)
        hidden2_errors = numpy.dot(self.wh1h2.T, hidden3_errors)
        hidden1_errors = numpy.dot(self.wih1.T, hidden2_errors)

        self.wh3o += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                         numpy.transpose(hidden3_outputs))

        self.wh2h3 += self.lr * numpy.dot((hidden3_errors * hidden3_outputs * (1.0 - hidden3_outputs)),
                                          numpy.transpose(hidden2_outputs))

        self.wh1h2 += self.lr * numpy.dot((hidden2_errors * hidden2_outputs * (1.0 - hidden2_outputs)),
                                          numpy.transpose(hidden1_outputs))

        self.wih1 += self.lr * numpy.dot((hidden1_errors * hidden1_outputs * (1.0 - hidden1_outputs)),
                                         numpy.transpose(inputs))

        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden1_inputs = numpy.dot(self.wih1, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        hidden3_inputs = numpy.dot(self.wh2h3, hidden2_outputs)
        hidden3_outputs = self.activation_function(hidden3_inputs)

        final_inputs = numpy.dot(self.wh3o, hidden3_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# 3 - h


input_nodes = 1024
hidden1_nodes = 1024
hidden2_nodes = 1024
hidden3_nodes = 1024
output_nodes = 2

learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden1_nodes, hidden2_nodes, hidden3_nodes, output_nodes, learning_rate)


def train_neural():
    with open('res_dict.json', 'r') as train_data_raw:
        train_data = json.load(train_data_raw)

    keys = list(train_data.keys())
    random.shuffle(keys)
    train_data_new = {}
    for key in keys:
        train_data_new[key] = train_data[key]
    train_list = []
    answers_list = []
    global_index = 0
    for key in keys:
        img = Image.open("data/"+key)
        train_list.append([])
        answers_list.append(train_data_new[key])
        global_index2 = 0
        for i in range(32):
            for j in range(32):
                #print(global_index, global_index2)
                train_list[global_index].append(sum(img.getpixel((i, j)))/3)
                global_index2 += 1
        global_index += 1

    epochs = 2
    for e in range(epochs):
        for i in range(len(train_list)):
            inputs = (numpy.asfarray(train_list[i]) / 255) - 0.01
            targets = []
            if answers_list[i] == 0:
                targets.append(0.99)
                targets.append(0.01)
            else:
                targets.append(0.01)
                targets.append(0.99)
            n.train(inputs, targets)

            pass
        print("circle: ", e + 1)
        pass
    print("training passed")


def save_weights():
    write_wih1 = open("weights/wih1.txt", "w")
    write_wh1h2 = open("weights/wh1h2.txt", "w")
    write_wh2h3 = open("weights/wh2h3.txt", "w")
    write_wh3o = open("weights/wh3o.txt", "w")

    what_to_write_wih1 = "], ", n.wih1.tolist().__repr__()[1:-1], ", ["
    what_to_write_wh1h2 = "], ", n.wh1h2.tolist().__repr__()[1:-1], ", ["
    what_to_write_wh2h3 = "], ", n.wh2h3.tolist().__repr__()[1:-1], ", ["
    what_to_write_wh3o = "], ", n.wh3o.tolist().__repr__()[1:-1], ", ["

    write_wih1.write("".join(what_to_write_wih1))
    write_wh1h2.write("".join(what_to_write_wh1h2))
    write_wh2h3.write("".join(what_to_write_wh2h3))
    write_wh3o.write("".join(what_to_write_wh3o))

    write_wih1.close()
    write_wh1h2.close()
    write_wh2h3.close()
    write_wh3o.close()


def update_weights():
    trained_wih1_file = open("weights/wih1.txt", 'r')
    trained_wh1h2_file = open("weights/wh1h2.txt", 'r')
    trained_wh2h3_file = open("weights/wh2h3.txt", 'r')
    trained_wh3o_file = open("weights/wh3o.txt", 'r')
    wh3o = trained_wh3o_file.readlines().__str__()
    wh2h3 = trained_wh2h3_file.readlines().__str__()
    wh1h2 = trained_wh1h2_file.readlines().__str__()
    wih1 = trained_wih1_file.readlines().__str__()

    wh3os = wh3o.split('], [')
    wh2h3s = wh2h3.split('], [')
    wh1h2s = wh1h2.split('], [')
    wih1s = wih1.split('], [')

    wh3os = numpy.matrix(list(map(lambda str: str.split(', '), wh3os))[1:-1]).astype(numpy.float)
    wh2h3s = numpy.matrix(list(map(lambda str: str.split(', '), wh2h3s))[1:-1]).astype(numpy.float)
    wh1h2s = numpy.matrix(list(map(lambda str: str.split(', '), wh1h2s))[1:-1]).astype(numpy.float)
    wih1s = numpy.matrix(list(map(lambda str: str.split(', '), wih1s))[1:-1]).astype(numpy.float)

    n.wh3o = wh3os
    n.wh2h3 = wh2h3s
    n.wh1h2 = wh1h2s
    n.wih1 = wih1s

    trained_wih1_file.close()
    trained_wh1h2_file.close()
    trained_wh2h3_file.close()
    trained_wh3o_file.close()


update_weights()

for i in range(260):
    train_list = []
    img = Image.open("photo/cutPixels"+str(i)+".jpg")
    for j in range(32):
        for k in range(32):
            # print(global_index, global_index2)
            train_list.append(sum(img.getpixel((j, k)))/3)
    print(n.query((numpy.asfarray(train_list) / 255) - 0.01))
    train_list.clear()
