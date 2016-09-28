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
