"""Usage: CNNFromScratch.py [-n N | --max_text_length=N] [-t T | --relu_threshold=T] [-h H | --host=H] [-p P | --port=P]
                [-w W | --weights_file_path=W]

Options:
    -n --max_text_length=N  Maximum length out of the input the CNN will read [default: 1014]
    -t --relu_threshold=T  The minimum threshold for the CNN activation function [default: 0.000001]
    -h --host=H  host server address [default: localhost]
    -p --port=P  port of host server [default: 8000]
    -w --weights_file_path=W  The absolute name & path for the model's saved weights file [default: ./CNNSavedWeights]
"""

import os
import docopt
import numpy as np
import pycnn as pc
from bottle import route, run, template, get, post, request

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
CNNFromScratch is a Pycnn implementation of "Text Understanding from Scratch" paper by Xiang Zhang & Yann LeCun
Published in arXiv in April 2016.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
arguments = docopt.docopt(__doc__)

QUANTIZATION_SIZE = 70
MAX_INPUT_SIZE = int(arguments['--max_text_length'])
relu_threshold = float(arguments['--relu_threshold'])
host = arguments['--host']
port = int(arguments['--port'])
weights_file_path = arguments['--weights_file_path']

print MAX_INPUT_SIZE + relu_threshold + port
templateArray = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '\n', '-', ',', ';',
                 '.', '!', '?', ':', '"', '\'', '/', '\\', '|', '_', '@', '#', '$', '%', '^', '&', '*', '~', '`',
                 '+', '-', '=', '<', '>', '(', ')', '[', ']', '{', '}']

id_to_label_dic = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: get_label
Input: label - a category name in th data set, labels_dictionary - dictionary with all previous categories.
Output: the label's id.
Functionality: function checks if the last seen label was previously seen, if not - add the label as a new category,
                and in any case returns the label id.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def get_label(label, labels_dictionary):
    if not label in labels_dictionary:
        labels_dictionary[label] = len(labels_dictionary)
    return labels_dictionary[label]


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: arrange_input
Input: input_table - a list of articles
Output: categories - a list of labels int id's instead of string, values - a list of concatenated strings combined
        of the article name and description, category_dic - a dictionary that maps id's and labels real string value.
Functionality: function go over all articles, concatenate all articles name and description, assign them to a values
                list, assign all labels id's to a categories list and creates the labels to id's dictionary.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def arrange_input(input_table):
    categories = []
    values = []
    category_dic = {}
    for line in input_table:
        split_line = line.split('","')
        category = get_label(split_line[0][1:], category_dic)
        title = split_line[1]
        description = split_line[2]
        categories.append(category)
        values.append(title + " " + description)
    return categories, values, category_dic


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: quantize_string
Input: string - a given article to quantize.
Output: a vector of the quantized article.
Functionality: for each letter in the sentence, the function creates an all zero Pycnn vector, if the letter is one
                of the 70 known letters it assigns the vector a 1-to-m proper coding (according to the letter
                location in the "templateArray", else keeps an all zero vector. after each letter the vector is
                concatenated to the output vector. Stops quantifying after "MAX_INPUT_SIZE" number of letters.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def quantize_string(string):
    quantized_vec = np.zeros(QUANTIZATION_SIZE * MAX_INPUT_SIZE)
    cur_index = 0
    for char in string:
        if cur_index/QUANTIZATION_SIZE >= MAX_INPUT_SIZE:
            break
        if char in templateArray:
            quantized_vec[templateArray.index(char) + cur_index] = 1
        cur_index += QUANTIZATION_SIZE
    quantized_vec_exp = pc.vecInput(QUANTIZATION_SIZE * MAX_INPUT_SIZE)
    quantized_vec_exp.set(quantized_vec)
    return quantized_vec_exp


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: initialize_cnn_layer_weights
Input: mode - Pycnn model, kernel_num - number of kernels in the current layer, kernel_size - size of kernel in the
        current layer, frame_size - input frame size from previous layer, layer_id_str - layer number.
Functionality: according to the mentioned above parameters the model creates the convolution layer weights, and it's
               bias vector weights, when the initialization value have gaussian distribution with a variance of 0.05.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def initialize_cnn_layer_weights(model, kernel_num, kernel_size, frame_size, layer_id_str):
    model.add_parameters(layer_id_str, (kernel_num, kernel_size * frame_size))\
        .load_array(np.random.normal(0, 0.05, (kernel_num, kernel_size * frame_size)))
    model.add_parameters("b" + layer_id_str, kernel_num).load_array(np.random.normal(0, 0.05, kernel_num))
    return


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: initialize_dense_layer_weights
Input: model - Pycnn model, cur_layer_dim - number of cells in this layer, pre_layer_dim - number of cells in
        previous layer, layer_id_str - layer number.
Functionality: according to the mentioned above parameters the model creates the dense layer weights, and it's
               bias vector weights, when the initialization value have gaussian distribution with a variance of 0.05.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def initialize_dense_layer_weights(model, cur_layer_dim, pre_layer_dim, layer_id_str):
    model.add_parameters(layer_id_str, (cur_layer_dim, pre_layer_dim))\
        .load_array(np.random.normal(0, 0.05, (cur_layer_dim, pre_layer_dim)))
    model.add_parameters("b" + layer_id_str, cur_layer_dim).load_array(np.random.normal(0, 0.05, cur_layer_dim))
    return


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: initialize_cnn_model_weights
Input: model - Pycnn model, dense_layers_dim - a list of integers representing the layer size, such that list size
        is number of such layers (final layer will represent the output layer), conv_layers_dims - same as
        dense_layers_dim, only with 2D list of (kernel number, kernel size) in each layer.
Functionality: supervising function to initialize all the model's weights according to the pre-given dimensions set
                by the user, and using the 2 initialization function mentioned above. also responsible of assigning
                each layer with her proper id number.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def initialize_cnn_model_weights(model, conv_layers_dims, dense_layers_dim):
    layer_id = 0
    prev_layer_dim = QUANTIZATION_SIZE
    for layer_dim in conv_layers_dims:
        initialize_cnn_layer_weights(model, layer_dim[0], layer_dim[1], prev_layer_dim, str(layer_id))
        prev_layer_dim = layer_dim[0]
        layer_id += 1

    # FLATTEN
    prev_layer_dim *= (MAX_INPUT_SIZE - 96) / 27

    for layer_dim in dense_layers_dim:
        initialize_dense_layer_weights(model, layer_dim, prev_layer_dim, str(layer_id))
        prev_layer_dim = layer_dim
        layer_id += 1
    return


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: calc_convolution_layer
Input: model - Pycnn model, layer_input - the current layer input vector, layer_id_str - current layer id,
        kernel_size - size of kernels in current layer, frame_size - frame size of the input in current layer,
        input_len - length of input in current layer.
Output: a Pycnn expression of the layer output vector after 1D convolution.
Functionality: this function is doing a 1D convolution over the given input, using kernels matrix and input vector
                multiplication.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def calc_convolution_layer(model, layer_input, layer_id_str, kernel_size, frame_size, input_len):
    output_vectors = []
    bias_vec = pc.parameter(model["b" + layer_id_str])
    conv_mat = pc.parameter(model[layer_id_str])
    flat_short_input_len = (input_len - kernel_size + 1) * frame_size
    for i in xrange(0, flat_short_input_len, frame_size):
        x = layer_input[i:i + kernel_size * frame_size]
        output_vectors.append(conv_mat * x + bias_vec)
    return pc.concatenate(output_vectors)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: calc_convolution_pooling_layer
Input: model - Pycnn model, layer_input - the current layer input vector, layer_id_str - current layer id,
        kernel_size - size of kernels in current layer, frame_size - frame size of the input in current layer,
        input_len - length of input in current layer, pooling_dim - current layer pooling window dimensions.
Output: a Pycnn expression of the layer output vector after 1D convolution and pooling.
Functionality: Same as previous function, with the addition of dimensionality reduction with max pooling.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def calc_convolution_pooling_layer(model, layer_input, layer_id_str, kernel_size, frame_size, input_len, pooling_dim):
    prepool_vectors = []
    output_vectors = []
    bias_vec = pc.parameter(model["b" + layer_id_str])
    conv_mat = pc.parameter(model[layer_id_str])
    flat_short_input_len = (input_len - kernel_size + 1) * frame_size
    for i in xrange(0, flat_short_input_len, frame_size):
        x = layer_input[i:i + kernel_size * frame_size]
        prepool_vectors.append(conv_mat * x + bias_vec)
        if (i + frame_size) % (pooling_dim * frame_size) == 0:
            output_vectors.append(pc.kmax_pooling(pc.concatenate_cols(prepool_vectors), 1))
            prepool_vectors = []
    return pc.concatenate(output_vectors)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: calc_dense_layer
Input: model - Pycnn model, layer_input - the current layer input vector, layer_id_str - current layer id.
Output: a Pycnn expression of the layer output vector after fully connected layer calculation.
Functionality: multiples input by all fully connected weights, and add the bias vector afterwards.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def calc_dense_layer(model, layer_input, layer_id_str):
    return pc.parameter(model[layer_id_str]) * layer_input + pc.parameter(model["b" + layer_id_str])


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: threshold
Input: input_vec - input vector, thresh - size of threshold number, vec_len - length of input vector.
Output: the input vector plus the threshold value on all vector cells.
Functionality: creates an equal sized vector with the threshold value in all his cells and add it to the input vector.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def threshold(input_vec, thresh, vec_len):
    threshold_nparr = np.zeros(vec_len) + thresh
    threshold_pcarr = pc.vecInput(vec_len)
    threshold_pcarr.set(threshold_nparr)
    return input_vec + threshold_pcarr


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: run_network
Input: model - Pycnn model, input_sentence - a given article, conv_layers_dims - convolution layers dimensions,
        dense_layers_dims - dense layers dimensions, thresh - threshold number.
Output: Pycnn vector expression with the value of the network forward propagation output.
Functionality: this function is the hurt of this code, and is representing the structure of the network. every layer
                in the origin paper is separated by an empty line and is assigned with its proper parameters.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def run_network(model, input_sentence, conv_layers_dims, dense_layers_dims, thresh):

    input_length = MAX_INPUT_SIZE
    quantized_sentence = quantize_string(input_sentence)

    conv_out = calc_convolution_pooling_layer(model, quantized_sentence, "0", conv_layers_dims[0][1], QUANTIZATION_SIZE, input_length, 3)
    input_length = (input_length - conv_layers_dims[0][1] + 1) / 3
    conv_out = pc.rectify(conv_out)
    conv_out = threshold(conv_out, thresh, conv_layers_dims[0][0] * input_length)

    conv_out = calc_convolution_pooling_layer(model, conv_out, "1", conv_layers_dims[1][1], conv_layers_dims[0][0], input_length, 3)
    input_length = (input_length - conv_layers_dims[1][1] + 1) / 3
    conv_out = pc.rectify(conv_out)
    conv_out = threshold(conv_out, thresh, conv_layers_dims[1][0] * input_length)

    conv_out = calc_convolution_layer(model, conv_out, "2", conv_layers_dims[2][1], conv_layers_dims[1][0], input_length)
    input_length -= conv_layers_dims[2][1] - 1
    conv_out = pc.rectify(conv_out)
    conv_out = threshold(conv_out, thresh, conv_layers_dims[2][0] * input_length)

    conv_out = calc_convolution_layer(model, conv_out, "3", conv_layers_dims[3][1], conv_layers_dims[2][0], input_length)
    input_length -= conv_layers_dims[3][1] - 1
    conv_out = pc.rectify(conv_out)
    conv_out = threshold(conv_out, thresh, conv_layers_dims[3][0] * input_length)

    conv_out = calc_convolution_layer(model, conv_out, "4", conv_layers_dims[4][1], conv_layers_dims[3][0], input_length)
    input_length -= conv_layers_dims[4][1] - 1
    conv_out = pc.rectify(conv_out)
    conv_out = threshold(conv_out, thresh, conv_layers_dims[4][0] * input_length)

    conv_out = calc_convolution_pooling_layer(model, conv_out, "5", conv_layers_dims[5][1], conv_layers_dims[4][0], input_length, 3)
    input_length = (input_length - conv_layers_dims[5][1] + 1) / 3
    conv_out = pc.rectify(conv_out)
    conv_out = threshold(conv_out, thresh, conv_layers_dims[5][0] * input_length)

    flattened_out = pc.reshape(conv_out, (conv_layers_dims[5][0] * (MAX_INPUT_SIZE - 96) / 27, 1))

    dense_out = calc_dense_layer(model, flattened_out, "6")
    dense_out = pc.rectify(dense_out)
    dense_out = threshold(dense_out, thresh, dense_layers_dims[0])
    dense_out = pc.dropout(dense_out, 0.5)

    dense_out = calc_dense_layer(model, dense_out, "7")
    dense_out = pc.rectify(dense_out)
    dense_out = threshold(dense_out, thresh, dense_layers_dims[1])
    dense_out = pc.dropout(dense_out, 0.5)

    dense_out = calc_dense_layer(model, dense_out, "8")
    return dense_out


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: predict_label
Input: model - Pycnn model, input_sentence - a given article, conv_layers_dims - convolution layers dimensions,
        dense_layers_dims - dense layers dimensions, threshold - threshold number.
Output: the predicted label of the network and the loss from th real one.
Functionality: same as previous function, with the addition of returnning the predicted label according to the
                highest value in the network output.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def predict_label(model, input_sentence, conv_layers_dims, dense_layers_dims, threshold):
    network_output = run_network(model, input_sentence, conv_layers_dims, dense_layers_dims, threshold)
    return np.argmax(network_output.value())


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: login
Output: an html format of the login screen with explanation of the page essence, text box and a "submit" button.
Functionality: returns an html format of the login screen with a text box to insert article to be classified and
                a "submit" button to send for examination.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
@get('/CNNFromScratch')
def login():
    return '''
        <form action="/CNNFromScratch" method="post">
            <p><b><font size='6'>NLP CNN From Scratch Website</font></b></p>
            <p>This web page is classifying articles to 4 categories (World, Sports, Business & Sci/Tech) according to
            <i>\'Text Understanding from Scratch\'</i> article, based on a 1D NLP convolutional neural network</p>
            <p>Link to original paper from X. Zhang & Y. LeCun:
            <a href="https://arxiv.org/pdf/1502.01710.pdf">Text Understanding from Scratch paper</a></p>
            <br>Please insert article text to the box bellow</br>
            <p>Article: <input name="article" type="text" size="200"/></p>
            <input value="Submit" type="submit"/>
        </form>
    '''

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: do_login
Output: an answer of which category is the most suited to the given text out of the given selection.
Functionality: in this function first a new model is being created, then the trained weights are loaded from the
                weights file, and lastly a forward propgation is being made, using the input text, and predict it's
                category.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
@post('/CNNFromScratch')
def do_login():
    article_to_label = request.forms.get('article')
    if len(article_to_label) < MAX_INPUT_SIZE:
        return "Text is to short to be classified (required minimal length of {})".format(MAX_INPUT_SIZE)
    if not os.path.exists(weights_file_path):
        return "No weights file was found, looked for file {}, in directory {}".format(weights_file_path, os.getcwd())

    model = pc.Model()
    cnn_layers_dimensions = [(256, 7), (256, 7), (256, 3), (256, 3), (256, 3), (256, 3)]
    dense_layers_dimensions = [1024, 1024, 4]
    initialize_cnn_model_weights(model, cnn_layers_dimensions, dense_layers_dimensions)
    model.load(weights_file_path)

    pc.renew_cg()
    predicted_label = predict_label(model, article_to_label, cnn_layers_dimensions, dense_layers_dimensions, relu_threshold)

    return "Article was classified to category: {}".format(id_to_label_dic[predicted_label])

# host server and port the page will run on
run(host=host, port=port)
