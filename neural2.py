import numpy
import scipy.special
import random
import json
from PIL import Image
from img_downloader import create_big_image, parse_map
import io



class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


input_nodes = 1024
hidden_nodes = 512
output_nodes = 2

learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


def save_neural():
    write_wih = open("weights/wih.txt", 'w')
    write_who = open("weights/who.txt", 'w')
    what_to_write_wih = "], ", n.wih.tolist().__repr__()[1:-1], ", ["
    what_to_write_who = "], ", n.who.tolist().__repr__()[1:-1], ", ["
    write_wih.write("".join(what_to_write_wih))
    write_who.write("".join(what_to_write_who))
    write_who.close()
    write_wih.close()

def update_weights():
    trained_wih_file = open("weights/wih.txt", 'r')
    trained_who_file = open("weights/who.txt", 'r')
    who = trained_who_file.readlines().__str__()
    wih = trained_wih_file.readlines().__str__()

    whos = who.split('], [')
    wihs = wih.split('], [')

    whos = numpy.matrix(list(map(lambda str: str.split(', '), whos))[1:-1]).astype(numpy.float)
    wihs = numpy.matrix(list(map(lambda str: str.split(', '), wihs))[1:-1]).astype(numpy.float)

    n.who = whos
    n.wih = wihs

    trained_wih_file.close()
    trained_who_file.close()


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


update_weights()


def query_tree(img):
    train_list = []
    for j in range(32):
        for k in range(32):
            # print(global_index, global_index2)
            train_list.append(sum(img.getpixel((j, k)))/3)
    res = n.query((numpy.asfarray(train_list) / 255) - 0.01)
    if res[0] > res[1]:
        return False
    else:
        return True


def sliding_window(x1, y1, x2, y2):
    img_raw = create_big_image(parse_map(x1, y1, x2, y2))
    img = Image.open(io.BytesIO(img_raw))
    new_im = Image.new('RGB', (img.size[0], img.size[1]))
    for x in range(0, img.size[0], 8):
        for y in range(0, img.size[1], 8):
            border = (x, y, x + 32, y + 32)  # left, up, right, bottom
            cropped2 = img.crop(border)
            if query_tree(cropped2):
                new_im.paste(cropped2, (x, y))
    new_im.save("test2.jpg")
    new_im.show()


sliding_window(104.640041, 52.934288, 104.690993, 52.927670)
