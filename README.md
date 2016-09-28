Convolutional Networks for Text Classification
==============================================

This repository contains all the work I did as my 3rd year final project at Bar-Ilan university, CS department.
I implemented the [Xiang Zhang](http://xlab.me.berkeley.edu/) and [Yann LeCun](http://yann.lecun.com/) articles: [Text Understanding from Scratch](https://arxiv.org/pdf/1502.01710.pdf) and [Character-level Convolutional Networks for Text
Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
on text classification on a charecter level using CNN.


## Project Discription
The goal of my project was to re-create and improve the work mentioned above using a 
 relatively "low level" artificial neural network language in-order to understand the 
 essance of the model. Which means to succeed and perform text classification task using 
 only character level vectors input method, rather then word vectors, which requires a 
 significant pre-process effort (to execute Word2Vec or a similar word embedding). For 
 that purpose a 9 layers convolutional neural network was used, in the attempt to achieve, 
 with a raw input of letters, a similar success to this the artificial vision is demonstrating 
 with a raw signal of pixels.

## External libraries
* Using [Pycnn](https://github.com/clab/cnn/pycnn) for the main focus network of this project.
* Using [Keras](https://github.com/fchollet/keras) for the secondary, more readable and easy to use network.
* Using [bottle](https://github.com/bottlepy/bottle) for creating a web page interface.
* Using [docopt](https://github.com/docopt/docopt) for creating an easy to use parameters access.

## Other requirements
To run the application, you will need to have [python 2.7](https://www.python.org/download/releases/2.7/) with all of libreries above (Keras is optional) installed in it's working directory.

## Running the project
The project Pycnn code is divided to 2 parts: <br />
1. [CNNFromScratch.py](https://github.com/AdamYaari/NLP_CNN_From_Scratch/blob/master/src/CNNFromScratch.py) - which contains the code needed to learn the model weights. <br />
2. [CNNFSWeb.py](https://github.com/AdamYaari/NLP_CNN_From_Scratch/blob/master/src/CNNFSWeb.py) - which supplies with the web page interface backend code to use the pre-trained weights to classify new article text. 

After fulfilling all of the requirmentes above, the code can be run with the following commands:
### part 1 
```
python2.7 /Path/to/file/directory/CNNFromScratch.py --cnn-mem 8192
```
With the following usage options:
```
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
```  

#### Code output
1. Weights file - which are saved every epoch (and overwrite the previous epoch weights which are redundant).
2. Tests log file - after every epoch a test is being made on the test set, and both the average loss and the test accuracy are writen to the log file, along with the run parameters.

### part 2
```
python2.7 /Path/to/file/directory/CNNFSWeb.py
```
With the following usage options:
```
-n --max_text_length=N  Maximum length out of the input the CNN will read [default: 1014]
-t --relu_threshold=T  The minimum threshold for the CNN activation function [default: 0.000001]
-h --host=H  host server address [default: localhost]
-p --port=P  port of host server [default: 8000]
-w --weights_file_path=W  The absolute name & path for the model's pre-trained saved weights file [default: ./CNNSavedWeights]
```

After running the above command the server is up and running and the web page can be accessed via any browser in the following address for the defualt host and port values:
```
http://localhost:8000/CNNFromScratch
```
Or in a more general fashion:
```
http://host_name:port_number/CNNFromScratch
```
To enjoy the newly trained weights, just add a text (at least 1014 charecters long) and press the "submit button".

#### Code output
the given text predicted category


### Pycnn Important Remarks
* In the current version of the Pycnn code there is no special adjustment for GPU adaptability, which might be neccesary if one wants to use a GPU to accelerate the run time.
* At the moment, with no GPU, each epoch takes approximatly 1 day (please take into consideration when setting epoch number.
* According to empiric results so far, in-order to get optimal results at least 8-10 epochs are required.
* --cnn-mem is defined as 8192 and as a required parameter to part 1 of the code, because of the default (128) batch size, this requirment can be reduced along with a reduction of the batch size (non is neccessary for batch size 8).

## Runnig the Keras code
Running the [KerasCNNFromScratch.py](https://github.com/AdamYaari/NLP_CNN_From_Scratch/blob/master/keras_src/KerasCNNFromScratch.py) is almost similar to part 1 of the Pycnn code, with the exception of "--cnn-mem" demand and a few minor differences in the optional parameters
```
python2.7 /Path/to/file/directory/KerasCNNFromScratch.py
```
With the following usage options:
```
    -n --max_text_length=N  Maximum length out of the input the CNN will read [default: 1014]
    -l --learning_rate=L  The CNN learning rate [default: 0.01]
    -m --momentum=M  The CNN momentum [default: 0.9]
    -t --relu_threshold=T  The minimum threshold for the CNN activation function [default: 0.000001]
    -b --batch_size=B  The CNN batch size [default: 128]
    -e --epoch_num=E  Number of epochs the CNN will execute 3 times with half the L in each round [default: 10]
    --weights_file_name=W  The absolute name & path for the model's saved weights file [default: ./CNNSavedWeights]
    --test_set_file_path=TES  The absolute name & path for the model's test set input file [default: ./AGNewsDB/test.csv]
    --train_set_file_path=TRS  The absolute name & path for the model's train set input file [default: ./AGNewsDB/train.csv]
```

### Keras Important Remarks
* Even without a GPU the keras model run a lot faster then it's Pycnn equivalent, and takes less then 2 days to finish a run of 10 epochs.
* I did not implemented a web interface that intract with the Keras saved weights, the one mentioned above is equiped solly for the Pycnn model.
* Due to it's relatively short run time, the keras model save it's weights only once in the end of the run.


#### Written by: Adam Yaari
##### Supervised by Dr. [Yoav Goldberg](https://www.cs.bgu.ac.il/~yoavg/uni/)

For any question, suggestion etc:

a.u.yaari@gmail.com
