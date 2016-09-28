"""Usage: CNNFromScratch.py [-n N | --max_text_length=N] [-l L | --learning_rate=L] [-m M | --momentum=M]
                [-t T | --relu_threshold=T] [-b B | --batch_size=B] [-e E | --epoch_num=E]
                [--weights_file_name=W][--test_set_file_path=TES] [--train_set_file_path=TRS]

Options:
    -n --max_text_length=N  Maximum length out of the input the CNN will read [default: 1014]
    -l --learning_rate=L  The CNN learning rate [default: 0.01]
    -m --momentum=M  The CNN momentum [default: 0.9]
    -t --relu_threshold=T  The minimum threshold for the CNN activation function [default: 0.000001]
    -b --batch_size=B  The CNN batch size [default: 128]
    -e --epoch_num=E  Number of epochs the CNN will execute 3 times with half the L in each round [default: 10]
    --weights_file_name=W  The absolute name & path for the model's saved weights file [default: ./CNNSavedWeights]
    --test_set_file_path=TES  The absolute name & path for the model's test set input file [default: ./AGNewsDB/test.csv]
    --train_set_file_path=TRS  The absolute name & path for the model's train set input file [default: ./AGNewsDB/train.csv]
"""

import docopt
from keras import backend as K
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras import initializations
from keras.layers.advanced_activations import ThresholdedReLU

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
KerasCNNFromScratch is a keras implementation of "Text Understanding from Scratch" paper by Xiang Zhang & Yann LeCun
Published in arXiv in April 2016.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
arguments = docopt.docopt(__doc__)

quantization_size = 70

templateArray = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '\n', '-', ',', ';',
                 '.', '!', '?', ':', '"', '\'', ' ', '/', '\\', '|', '_', '@', '#', '$', '%', '^', '&', '*', '~', '`',
                 '+', '-', '=', '<', '>', '(', ')', '[', ']', '{', '}']

# initialization of the network parameters
maxInput_size = int(arguments['--max_text_length'])
learning_rate = float(arguments['--learning_rate'])
momentum = float(arguments['--momentum'])
K._EPSILON = float(arguments['--relu_threshold'])
batch_size = int(arguments['--batch_size'])
epoch_num = int(arguments['--epoch_num'])
weights_file_name = arguments['--weights_file_name']
test_set_file_path = arguments['--test_set_file_path']
train_set_file_path = arguments['--train_set_file_path']


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: gaussian_init
Input: shape - the shape of weights matrix to be initialized.
Output: weights in required shape, initialized with gaussian variance of 0.05.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def gaussian_init(shape, name=None):
    return initializations.normal(shape, scale=0.05, name=name)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: arrange_input
Input: input_table - a list of articles
Output: categories - a list of labels int id's instead of string, values - a list of concatenated strings combined
        of the article name and description.
Functionality: function go over all articles, concatenate all articles name and description, assign them to a values
                list and assign all labels id's to a categories list.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def arrange_input(input_table):
    categories = []
    values = []
    for line in input_table:
        split_line = line.split('","')
        category = split_line[0][1:]
        title = split_line[1]
        description = split_line[2]
        categories.append(category)
        values.append(title + " " + description)
    return categories, values


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: quantize_character
Input: char - a letter to be quantized.
Output: the letter quantized vector.
Functionality: the function creates an all zero numpy vector and if the letter is one of the 70 known letters it
                assigns the vector a 1-to-m proper coding (according to the letter location in the "templateArray",
                else keeps an all zero vector.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def quantize_character(char):
    quantized_char = np.zeros(quantization_size)
    if char in templateArray:
        quantized_char[templateArray.index(char)] = 1
    return quantized_char.transpose()


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: quantize_values
Input: input_array - array of articles strings.
Output: all articles as quantized matrix.
Functionality: for each article in the input array - quantize each letter, concatenate it to a matrix and concatenate
                it to a 3D matrix of all quantized articles.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def quantize_values(input_array):
    quantized_input = np.zeros((len(input_array), maxInput_size, quantization_size))
    for i, article in enumerate(input_array):
        for j, char in enumerate(article):
            quantized_input[i, j] = quantize_character(char)
            if j == maxInput_size - 1:
                break
        if i % 10000 == 0 and i > 0:
            print("{} articles quantized".format(i))
    print("{} articles quantized".format(len(quantized_input)))
    return quantized_input


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: quantize_categories
Input: input_array - array of articles labels, cat_dim - number of categories in total.
Output: all articles labels quantized matrix.
Functionality: create a "One Hot Vector" coding for each label number, when the "1" location is determined according
                to label value.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def quantize_categories(input_array, cat_dim):
    quantized_input = np.zeros((len(input_array), cat_dim))
    for i, article_cat in enumerate(input_array):
        quantized_input[i, int(article_cat) - 1] = 1
        if i % 10000 == 0 and i > 0:
            print("{} categories quantized".format(i))
    print("{} categories quantized".format(len(quantized_input)))
    return quantized_input

######################################################################################################################
print('Opening input files...')
test_input = open(test_set_file_path, 'r')
train_input = open(train_set_file_path, 'r')

print('Arranging input to correct format...')
train_categories_int, train_strings = arrange_input(train_input)
test_categories_int, test_strings = arrange_input(test_input)
num_of_cat = len(set(train_categories_int))

print('Executing train set quantization...')
train_values = quantize_values(train_strings)
train_labels = quantize_categories(train_categories_int, num_of_cat)

print('Executing test set quantization...')
test_values = quantize_values(test_strings)
test_labels = quantize_categories(test_categories_int, num_of_cat)
print('Values and labels have been successfully formatted')

sgd = SGD(lr=learning_rate, momentum=momentum, decay=1e-5)

inputs = Input(shape=(maxInput_size, quantization_size))

# Keras model to the CNN from scratch
x = Convolution1D(256, 7, init=gaussian_init)(inputs)
x = ThresholdedReLU(theta=1e-6,)(x)
x = MaxPooling1D(pool_length=3, stride=3, border_mode='valid')(x)
x = Convolution1D(256, 7, init=gaussian_init)(x)
x = ThresholdedReLU(theta=1e-6)(x)
x = MaxPooling1D(pool_length=3, stride=3, border_mode='valid')(x)
x = Convolution1D(256, 3, init=gaussian_init)(x)
x = ThresholdedReLU(theta=1e-6)(x)
x = Convolution1D(256, 3, init=gaussian_init)(x)
x = ThresholdedReLU(theta=1e-6)(x)
x = Convolution1D(256, 3, init=gaussian_init)(x)
x = ThresholdedReLU(theta=1e-6)(x)
x = Convolution1D(256, 3, init=gaussian_init)(x)
x = ThresholdedReLU(theta=1e-6)(x)
x = MaxPooling1D(pool_length=3, stride=3, border_mode='valid')(x)

flt = Flatten()(x)

fc1 = Dense(1024, init=gaussian_init)(flt)
fc1 = ThresholdedReLU(theta=1e-6)(fc1)
do1 = Dropout(0.5)(fc1)
fc2 = Dense(1024, init=gaussian_init)(do1)
fc2 = ThresholdedReLU(theta=1e-6)(fc2)
do2 = Dropout(0.5)(fc2)
predict = Dense(num_of_cat, activation='softmax', init=gaussian_init)(do2)

model = Model(input=inputs, output=predict)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

print('Fitting...')

# train epoch number with half the learning rate 3 times every epoch run
for iteration in range(0, 3):
    if iteration >= 1:
        cur_lr = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(cur_lr / 2)

    model.fit(train_values, train_labels, batch_size=batch_size, nb_epoch=epoch_num, validation_data=(test_values, test_labels),
              verbose=1)  # starts training

model.save(weights_file_name)
