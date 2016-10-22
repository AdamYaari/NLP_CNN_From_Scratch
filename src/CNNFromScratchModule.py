import numpy as np
import pycnn as pc

QUANTIZATION_SIZE = 70
MAX_INPUT_SIZE = 1014
templateArray = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '\n', '-', ',', ';',
                 '.', '!', '?', ':', '"', '\'', '/', '\\', '|', '_', '@', '#', '$', '%', '^', '&', '*', '~', '`',
                 '+', '-', '=', '<', '>', '(', ')', '[', ']', '{', '}']
cnn_layers_dimensions = [(256, 7), (256, 7), (256, 3), (256, 3), (256, 3), (256, 3)]
dense_layers_dimensions = [1024, 1024, 4]

'''Function checks if the last seen label was previously seen, if not - add the label as a new category,
                and in any case returns the label id.

:param label: a category name in th data set
:param labels_dictionary: dictionary with all previous categories
:type label: string
:type labels_dictionary: dictionary<string,int>
:returns: the label's id
:rtype: int
'''
def get_label(label, labels_dictionary):
    if not label in labels_dictionary:
        labels_dictionary[label] = len(labels_dictionary)
    return labels_dictionary[label]


'''Function go over all articles, concatenate all articles name and description, assign them to a values
                list, assign all labels id's to a categories list and creates the labels to id's dictionary.

:param input_table: a list of articles
:type input_table: list<string>
:returns categories: a list of labels id's instead of string
:returns values: a list of concatenated strings combined of the article name and description
:returns category_dic: a dictionary that maps id's and labels real string value
:rtype categories: list<int>
:rtype values: list<string>
:rtype category_dic: dictionary<string,int>
'''
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


'''For each letter in the sentence, the function creates an all zero Pycnn vector, if the letter is one
                of the 70 known letters it assigns the vector a 1-to-m proper coding (according to the letter
                location in the "templateArray", else keeps an all zero vector. after each letter the vector is
                concatenated to the output vector. Stops quantifying after "MAX_INPUT_SIZE" number of letters.

:param string: a given article to quantize
:type string: string
:returns: a vector of the quantized article.
:rtype: Pycnn vector expression
'''
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


'''According to the mentioned above parameters the model creates the convolution layer weights, and it's
               bias vector weights, when the initialization value have gaussian distribution with a variance of 0.05.

:param model: Pycnn model
:param kernel_num: number of kernels in the current layer
:param kernel_size: size of kernel in the current layer
:param frame_size: input frame size from previous layer
:param layer_id_str: layer number
:type model: Pycnn model
:type kernel_num: int
:type kernel_size: int
:type frame_size: int
:type layer_id_str: int
'''
def initialize_cnn_layer_weights(model, kernel_num, kernel_size, frame_size, layer_id_str):
    model.add_parameters(layer_id_str, (kernel_num, kernel_size * frame_size))\
        .load_array(np.random.normal(0, 0.05, (kernel_num, kernel_size * frame_size)))
    model.add_parameters("b" + layer_id_str, kernel_num).load_array(np.random.normal(0, 0.05, kernel_num))
    return


'''According to the mentioned above parameters the model creates the dense layer weights, and it's
               bias vector weights, when the initialization value have gaussian distribution with a variance of 0.05.

:param model: Pycnn model
:param cur_layer_dim: number of cells in this layer
:param pre_layer_dim: number of cells in previous layer
:param layer_id_str: layer number
:type model: Pycnn model
:type cur_layer_dim: int
:type pre_layer_dim: int
:type layer_id_str: int
'''
def initialize_dense_layer_weights(model, cur_layer_dim, pre_layer_dim, layer_id_str):
    model.add_parameters(layer_id_str, (cur_layer_dim, pre_layer_dim))\
        .load_array(np.random.normal(0, 0.05, (cur_layer_dim, pre_layer_dim)))
    model.add_parameters("b" + layer_id_str, cur_layer_dim).load_array(np.random.normal(0, 0.05, cur_layer_dim))
    return


'''Supervising function to initialize all the model's weights according to the pre-given dimensions set
                by the user, and using the 2 initialization function mentioned above. also responsible of assigning
                each layer with her proper id number.

:param model: Pycnn model
:param dense_layers_dim: a list of integers representing the layer size, such that list size is number of such layers
                            (final layer will represent the output layer)
:param conv_layers_dims: same as dense_layers_dim, only with 2D list of (kernel number, kernel size) in each layer
:param layer_id_str: layer number
:type model: Pycnn model
:type dense_layers_dim: list<tuple<int>>
:type conv_layers_dims: list<tuple<int>>
'''
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


'''Executing a 1D convolution over the given input, using kernels matrix and input vector
                multiplication.

:param model: Pycnn model
:param layer_input: the current layer input vector
:param layer_id_str: current layer id
:param kernel_size: size of kernels in current layer
:param frame_size: frame size of the input in current layer
:param input_len: length of input in current layer
:type model: Pycnn model
:type layer_input: Pycnn vector expression
:type layer_id_str: string
:type kernel_size: int
:type frame_size: int
:type input_len: int
:returns: a Pycnn expression of the layer output vector after 1D convolution.
:rtype: Pycnn vector expression
'''
def calc_convolution_layer(model, layer_input, layer_id_str, kernel_size, frame_size, input_len):
    output_vectors = []
    bias_vec = pc.parameter(model["b" + layer_id_str])
    conv_mat = pc.parameter(model[layer_id_str])
    flat_short_input_len = (input_len - kernel_size + 1) * frame_size
    for i in xrange(0, flat_short_input_len, frame_size):
        x = layer_input[i:i + kernel_size * frame_size]
        output_vectors.append(conv_mat * x + bias_vec)
    return pc.concatenate(output_vectors)


'''Same as previous function, with the addition of dimensionality reduction with max pooling.

:param model: Pycnn model
:param layer_input: the current layer input vector
:param layer_id_str: current layer id
:param kernel_size: size of kernels in current layer
:param frame_size: frame size of the input in current layer
:param input_len: length of input in current layer
:param pooling_dim: current layer pooling window dimensions
:type model: Pycnn model
:type layer_input: Pycnn vector expression
:type layer_id_str: string
:type kernel_size: int
:type frame_size: int
:type input_len: int
:type pooling_dim: int
:returns: a Pycnn expression of the layer output vector after 1D convolution and pooling.
:rtype: Pycnn vector expression
'''
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


'''Multiples input by all fully connected weights, and add the bias vector afterwards.

:param model: Pycnn model
:param layer_input: the current layer input vector
:param layer_id_str: current layer id
:type model: Pycnn model
:type layer_input: Pycnn vector expression
:type layer_id_str: string
:returns: a Pycnn expression of the layer output vector after fully connected layer calculation.
:rtype: Pycnn vector expression
'''
def calc_dense_layer(model, layer_input, layer_id_str):
    return pc.parameter(model[layer_id_str]) * layer_input + pc.parameter(model["b" + layer_id_str])


'''Creates an equal sized vector with the threshold value in all his cells and add it to the input vector.

:param input_vec: input vector
:param thresh: size of threshold number
:param vec_len: length of input vector
:type input_vec: Pycnn vector expression
:type thresh: float
:type vec_len: int
:returns: the input vector plus the threshold value on all vector cells.
:rtype: Pycnn vector expression
'''
def threshold(input_vec, thresh, vec_len):
    threshold_nparr = np.zeros(vec_len) + thresh
    threshold_pcarr = pc.vecInput(vec_len)
    threshold_pcarr.set(threshold_nparr)
    return input_vec + threshold_pcarr


'''This function is the hurt of this code, and is representing the structure of the network. every layer
                in the origin paper is separated by an empty line and is assigned with its proper parameters.

:param model: Pycnn model
:param input_sentence: a given article
:param conv_layers_dims: convolution layers dimensions
:param dense_layers_dims: dense layers dimensions
:param thresh: threshold number
:type model: Pycnn model
:type input_sentence: string
:type conv_layers_dims: list<tuple<int>>
:type dense_layers_dims: list<tuple<int>>
:type thresh: float
:returns: Pycnn vector expression with the value of the network forward propagation output.
:rtype: Pycnn vector expression
'''
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


'''Function is using the "run_network" method and is calculating its loss according to the true
                label of the article.

:param model: Pycnn model
:param input_sentence: a given article
:param conv_layers_dims: convolution layers dimensions
:param dense_layers_dims: dense layers dimensions
:param thresh: threshold number
:type model: Pycnn model
:type input_sentence: string
:type conv_layers_dims: list<tuple<int>>
:type dense_layers_dims: list<tuple<int>>
:type thresh: float
:returns: network loss for a given article
:rtype: Pycnn float expression
'''
def get_loss(model, input_sentence, label, conv_layers_dims, dense_layers_dims, threshold):
    network_output = run_network(model, input_sentence, conv_layers_dims, dense_layers_dims, threshold)
    return pc.pickneglogsoftmax(network_output, label)


'''Function is using the "run_network" method, calculating its loss according to the true
                label of the article and returning the predicted label according to the
                highest value in the network output.

:param model: Pycnn model
:param input_sentence: a given article
:param conv_layers_dims: convolution layers dimensions
:param dense_layers_dims: dense layers dimensions
:param thresh: threshold number
:type model: Pycnn model
:type input_sentence: string
:type conv_layers_dims: list<tuple<int>>
:type dense_layers_dims: list<tuple<int>>
:type thresh: float
:returns: predicted label of the network, network loss for a given article
:rtype: int, Pycnn float expression
'''
def predict_label(model, input_sentence, label, conv_layers_dims, dense_layers_dims, threshold):
    network_output = run_network(model, input_sentence, conv_layers_dims, dense_layers_dims, threshold)
    return np.argmax(network_output.value()), pc.pickneglogsoftmax(network_output, label).value()


'''This function run the training of the network weights on the training set, with high verbosity for
                train and test time to keep track of it's progress. Every batch size number of samples the back
                propagation and the weights update takes place, every 10 batches a status print take place, and
                every epoch a test and a weights save takes place.

:param model: Pycnn model
:param trainer: pre initialized Pycnn trainer
:param batch_size: size of batch
:param epoch_num: number of epochs to run
:param train_inputs: all train set values
:param train_labels: all train set labels
:param test_inputs: all test set values
:param test_labels: all test set labels
:param cnn_dims: convolution layers dimensions
:param dense_dims: dense layers dimensions
:param threshold: threshold number
:param test_file_name: test results file name
:param weights_file_name: saved weights file name
:type model: Pycnn model
:type trainer: Pycnn trainer
:type batch_size: int
:type epoch_num: int
:type train_inputs: list<string>
:type train_labels: list<int>
:type test_inputs: list<string>
:type test_labels: list<int>
:type cnn_dims: list<tuple<int>>
:type dense_dims: list<tuple<int>>
:type threshold: float
:type test_file_name: string
:type weights_file_name: string
'''
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


'''This function test the networks weights success on the test set, according to number of successful
                runs of predicting the right label via forward propagation, and acoording to avrage loss value at
                the end of the test.

:param model: Pycnn model
:param test_inputs: all test set values
:param test_labels: all test set labels
:param cnn_dims: convolution layers dimensions
:param dense_dims: dense layers dimensions
:param threshold: threshold number
:param file_name: test results file name
:type model: Pycnn model
:type test_inputs: list<string>
:type test_labels: list<int>
:type cnn_dims: list<tuple<int>>
:type dense_dims: list<tuple<int>>
:type threshold: float
:type file_name: string
'''
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
