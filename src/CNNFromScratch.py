"""Usage: CNNFromScratch.py [-n N | --max_text_length=N] [-l L | --learning_rate=L] [-m M | --momentum=M]
                [-r R | --regularization=R] [-t T | --relu_threshold=T] [-b B | --batch_size=B] [-e E | --epoch_num=E]
                [--weights_file_name=W] [--test_res_file_name=F] [--test_set_file_path=TES] [--train_set_file_path=TRS]

Options:
    -n --max_text_length=N  Maximum length out of the input the CNN will read [default: 1014]
    -l --learning_rate=L  The CNN learning rate [default: 0.01]
    -m --momentum=M  The CNN momentum [default: 0.9]
    -r --regularization=R  The CNN regularization constant [default: 0.000001]
    -t --relu_threshold=T  The minimum threshold for the CNN activation function [default: 0.000001]
    -b --batch_size=B  The CNN batch size [default: 128]
    -e --epoch_num=E  Number of epochs the CNN will execute [default: 10]
    --weights_file_name=W  The absolute name & path for the model's saved weights file [default: ./CNNSavedWeights]
    --test_res_file_name=F  The absolute name & path for the model's test results output file [default: ./CNNTestResults.txt]
    --test_set_file_path=TES  The absolute name & path for the model's test set input file [default: ./AGNewsDB/test.csv]
    --train_set_file_path=TRS  The absolute name & path for the model's train set input file [default: ./AGNewsDB/train.csv]
"""

import numpy as np
import pycnn as pc
import docopt

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
CNNFromScratch is a Pycnn implementation of "Text Understanding from Scratch" paper by Xiang Zhang & Yann LeCun
Published in arXiv in April 2016.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
arguments = docopt.docopt(__doc__)

QUANTIZATION_SIZE = 70

templateArray = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '\n', '-', ',', ';',
                 '.', '!', '?', ':', '"', '\'', '/', '\\', '|', '_', '@', '#', '$', '%', '^', '&', '*', '~', '`',
                 '+', '-', '=', '<', '>', '(', ')', '[', ']', '{', '}']

# initialization of the network parameters
MAX_INPUT_SIZE = int(arguments['--max_text_length'])
learning_rate = float(arguments['--learning_rate'])
momentum = float(arguments['--momentum'])
regularization = float(arguments['--regularization'])
relu_threshold = float(arguments['--relu_threshold'])
batch_size = int(arguments['--batch_size'])
epoch_num = int(arguments['--epoch_num'])
weights_file_name = arguments['--weights_file_name']
test_res_file_name = arguments['--test_res_file_name']
test_set_file_path = arguments['--test_set_file_path']
train_set_file_path = arguments['--train_set_file_path']

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
Function Name: get_loss
Input: model - Pycnn model, input_sentence - a given article, conv_layers_dims - convolution layers dimensions,
        dense_layers_dims - dense layers dimensions, threshold - threshold number.
Output: network loss for a given article.
Functionality: function is using the above "run_network" method and is calculating its loss according to the true
                label of the article.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def get_loss(model, input_sentence, label, conv_layers_dims, dense_layers_dims, threshold):
    network_output = run_network(model, input_sentence, conv_layers_dims, dense_layers_dims, threshold)
    return pc.pickneglogsoftmax(network_output, label)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: predict_label
Input: model - Pycnn model, input_sentence - a given article, conv_layers_dims - convolution layers dimensions,
        dense_layers_dims - dense layers dimensions, threshold - threshold number.
Output: the predicted label of the network and the loss from th real one.
Functionality: same as previous function, with the addition of returnning the predicted label according to the
                highest value in the network output.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def predict_label(model, input_sentence, label, conv_layers_dims, dense_layers_dims, threshold):
    network_output = run_network(model, input_sentence, conv_layers_dims, dense_layers_dims, threshold)
    return np.argmax(network_output.value()), pc.pickneglogsoftmax(network_output, label).value()


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: train
Input: model - Pycnn model, trainer - pre initialized Pycnn trainer, batch_size - size of batch, epoch_num - number
        of epochs to run, train_inputs - all train set values, train_labels - all train set labels, test_inputs -
        all test set values, test_labels - all test set labels, cnn_dims - convolution layers dimensions,
        dense_dims - dense layers dimensions, threshold - threshold number, test_file_name - test results file name,
        weights_file_name - saved weights file name.
Functionality: this function run the training of the network weights on the training set, with high verbosity for
                train and test time to keep track of it's progress. Every batch size number of samples the back
                propagation and the weights update takes place, every 10 batches a status print take place, and
                every epoch a test and a weights save takes place.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def train(model, trainer, batch_size, epoch_num, train_inputs, train_labels, test_inputs, test_labels,
          cnn_dims, dense_dims, threshold, test_file_name, weights_file_name):
    for epoch in xrange(epoch_num):
        disp_batch_loss = 0
        batch_loss_list = []
        index = 1
        input_data = zip(train_inputs, train_labels)
        print "Started epoch #{}".format(epoch)
        pc.renew_cg()
        for sentence, label in input_data:
            loss = get_loss(model, sentence, label, cnn_dims, dense_dims, threshold)
            batch_loss_list.append(loss)
            if index % batch_size == 0:
                batch_loss = pc.average(batch_loss_list)
                batch_loss_val = batch_loss.value()
                disp_batch_loss += batch_loss_val
                batch_loss.backward()
                trainer.update()
                batch_loss_list = []
                pc.renew_cg()

            if index % (10 * batch_size) == 0:
                print "Epoch # {}, samples # {}, loss: {}".format(epoch, index, disp_batch_loss/10)
                disp_batch_loss = 0
            index += 1
        print "End of epoch #{}".format(epoch)
        print "Saving weights..."
        model.save(weights_file_name)
        print "Weights are saved to: {}".format(weights_file_name)
        print "Testing accuracy..."
        test(model, test_inputs, test_labels, cnn_dims, dense_dims, threshold, test_file_name)
    return


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Function Name: test
Input: model - Pycnn model, test_inputs - all test set values, test_labels - all test set labels, cnn_dims -
        convolution layers dimensions, dense_dims - dense layers dimensions, threshold - threshold number,
        file_name - test results file name.
Functionality: this function test the networks weights success on the test set, according to number of successful
                runs of predicting the right label via forward propagation, and acoording to avrage loss value at
                the end of the test.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def test(model, test_inputs, test_labels, cnn_dims, dense_dims, threshold, file_name):
    input_data = zip(test_inputs, test_labels)
    num_of_success = 0
    total_loss = 0
    index = 0
    pc.renew_cg()
    for sentence, label in input_data:
        predicted_label, loss = predict_label(model, sentence, label, cnn_dims, dense_dims, threshold)
        total_loss += loss

        if predicted_label == label:
            num_of_success += 1

        pc.renew_cg()
        index += 1

    test_loss = total_loss / float(len(input_data))
    test_accuracy = (100 * num_of_success) / float(len(input_data))
    print "Test is done!"
    print "Average test loss: {}".format(test_loss)
    print "Test accuracy: {}%\n".format(test_accuracy)
    with open(file_name, "ab") as text_file:
        text_file.write("\nAverage test loss: {}, Test accuracy: {}%".format(test_loss, test_accuracy))
    print "Test results were saved in file: {}".format(file_name)
    return
######################################################################################################################

print 'Opening input files...'
test_input = open(test_set_file_path, 'r')
train_input = open(train_set_file_path, 'r')

print 'Arranging input to correct format...'
train_categories_int, train_strings, labels_dic = arrange_input(train_input)
test_categories_int, test_strings, redundant_labels_dic = arrange_input(test_input)
num_of_cat = len(set(train_categories_int))
print 'Values and labels have been successfully formatted'

# initialization of network layers dimensions
cnn_layers_dimensions = [(256, 7), (256, 7), (256, 3), (256, 3), (256, 3), (256, 3)]
dense_layers_dimensions = [1024, 1024, 4]

model = pc.Model()

initialize_cnn_model_weights(model, cnn_layers_dimensions, dense_layers_dimensions)

SGD_trainer = pc.MomentumSGDTrainer(model, e0=learning_rate, mom=momentum)

print "Strat training with values: \nLearning rate: {} \nMomentum: {} \nRegularization constant: {} \nBatch size: {} " \
      "\nEpoch number: {}".format(learning_rate, momentum, regularization, batch_size, epoch_num)

with open(test_res_file_name, "w") as text_file:
    text_file.write("Test results for: \nlr: {} \nmom: {} \nregular: {} \nthresh: {} \nbatch: {} \nepochs: {}"
                    .format(learning_rate, momentum, regularization, relu_threshold, batch_size, epoch_num))

# train and test of the network
train(model, SGD_trainer, batch_size, epoch_num, train_strings, train_categories_int, test_strings, test_categories_int,
      cnn_layers_dimensions, dense_layers_dimensions, relu_threshold, test_res_file_name, weights_file_name)

print "FINISHED TRAINING SUCCESSFULLY"



