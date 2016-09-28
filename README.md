Convolutional Networks for Text Classification
==============================================

This repository contains all the work I did as my 3rd year final project at Bar-Ilan university, CS department.
I implemented the [Xiang Zhang](http://xlab.me.berkeley.edu/) and [Yann LeCun](http://yann.lecun.com/) articles: [Text Understanding from Scratch](https://arxiv.org/pdf/1502.01710.pdf) and [Character-level Convolutional Networks for Text
Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
on Random Features which scales-up kernel methods.


## Project Discription
The goal of our project was identifying phonemes from the TIMIT dataset,
 that is, given an unknown phoneme, classifying it as one of 39 possible phonemes
  (reduced set used instead of 44). Therefore, we had to use a classification 
  algorithm for a multi-class problem. We have used the SVM algorithm since 
  it solves a large-margin problem. Using the Random Features kernel of 
  Ali Rahimi and Ben Rech (which their article was mentioned above). 
