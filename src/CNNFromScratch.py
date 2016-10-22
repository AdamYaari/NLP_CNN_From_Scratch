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

import pycnn as pc
import docopt
import CNNFromScratchModule as module

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
CNNFromScratch is a Pycnn implementation of "Text Understanding from Scratch" paper by Xiang Zhang & Yann LeCun
Published in arXiv in April 2016.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
arguments = docopt.docopt(__doc__)

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


print 'Opening input files...'
test_input = open(test_set_file_path, 'r')
train_input = open(train_set_file_path, 'r')

print 'Arranging input to correct format...'
train_categories_int, train_strings, labels_dic = module.arrange_input(train_input)
test_categories_int, test_strings, redundant_labels_dic = module.arrange_input(test_input)
num_of_cat = len(set(train_categories_int))
print 'Values and labels have been successfully formatted'

model = pc.Model()

module.initialize_cnn_model_weights(model, module.cnn_layers_dimensions, module.dense_layers_dimensions)

SGD_trainer = pc.MomentumSGDTrainer(model, e0=learning_rate, mom=momentum)

print "Strat training with values: \nLearning rate: {} \nMomentum: {} \nRegularization constant: {} \nBatch size: {} " \
      "\nEpoch number: {}".format(learning_rate, momentum, regularization, batch_size, epoch_num)

with open(test_res_file_name, "w") as text_file:
    text_file.write("Test results for: \nlr: {} \nmom: {} \nregular: {} \nthresh: {} \nbatch: {} \nepochs: {}"
                    .format(learning_rate, momentum, regularization, relu_threshold, batch_size, epoch_num))

# train and test of the network
module.train(model, SGD_trainer, batch_size, epoch_num, train_strings, train_categories_int, test_strings,
             test_categories_int, module.cnn_layers_dimensions, module.dense_layers_dimensions, relu_threshold,
             test_res_file_name, weights_file_name)

print "FINISHED TRAINING SUCCESSFULLY"
