# asynchronous-dql-theano

## Description
Theano implementation of one step Q-learning:
[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783v2.pdf)

## System Requirements
Theano and Arcade-Learning-Environement

## Installation
* Theano:
	[website](http://deeplearning.net/software/theano/install.html)
    pip install theano
 
* Arcade learning environment :

    git clone https://github.com/mgbellemare/Arcade-Learning-Environment
    cd Arcade-Learning-Environment
    mkdir buid
    cd build
    cmake -DUSE_SDL=OFF -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF ..
    make
    cd ..
    pip install .


## Running
    python trainAgent.py breakout.bin
