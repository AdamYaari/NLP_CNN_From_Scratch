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
The project Pycnn code is divided to 2 parts:
* [CNNFromScratch.py](https://github.com/AdamYaari/NLP_CNN_From_Scratch/blob/master/src/CNNFromScratch.py) - which contains the code needed to learn the model weights.
* [CNNFSWeb.py](https://github.com/AdamYaari/NLP_CNN_From_Scratch/blob/master/src/CNNFSWeb.py) - which supplies with the web page interface backend code to use the pre-trained weights to classify new article text.

After fulfilling all of the requirmentes above, the code can be run with the following commands:
* part 1 - 
```
python2.7 /Path/to/file/directory/CNNFromScratch.py --cnn-mem 8192
```
With the following usage options:
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
    
* part 2 - 
```
python2.7 /Path/to/file/directory/CNNFSWeb.py
```
With the following usage options:
    -n --max_text_length=N  Maximum length out of the input the CNN will read [default: 1014]
    -t --relu_threshold=T  The minimum threshold for the CNN activation function [default: 0.000001]
    -h --host=H  host server address [default: localhost]
    -p --port=P  port of host server [default: 8000]
    -w --weights_file_path=W  The absolute name & path for the model's saved weights file [default: ./CNNSavedWeights]
