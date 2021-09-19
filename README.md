# Neural Network from Scratch
Building a Neural Network for UNIBE CAS DS 2021

#### Contributers:
- Maria Mumtaz, maria.mumtaz@students.unibe.ch
- Davi Bicudo, davi.bicudo@students.unibe.ch

## Description

In this repository, a fully connected 5 layers Neural Network is build. Then Neural Network is then used to recognize Images in mnist dataset(hand-written digits)
and cvnc dataset(Cat vs. No Cat)

## Technologies used

The neural Network is build from scratch. NeuralNetExampleFinal.ipynb has it made with libraries like numpy and plain python code. It trains the model and uses gradient
decent to compute the costs.
The model is then compared to scikit-learn's Multi-Layer Perceptron to evaluate the quality of our Model

## Structure


- src: All python source Files
- model: Saved models
- data: mnist and cvnc test/train sets
- images: Test image

## Source directiry
- activation: has activation all activation functions and their derivatives (e.g sigmoid, sigmoid_prime)
- loss: computes the losses
- config: sets up seed, number of classes, number of epochs, learning rate
- util: has utility functions like dump_model, load_model
- data_loader: loads the data and returns train and test sets
- nn_unit: abstract class for any nerual network unit
- fc_unit: implementation of nn_unit (has forward and backward propagation functions, and calculates weights)
- nn_layer: abstract class for neural network layer
- fc_layer : Self implementation of a fully coonected layer
- nn_network: adds layers, trains model, computes costs, predicts and evaluate


## Results

- For the cvnc dataset, Minimal cost achieved is 0.00073 with an accuracy of 76%
- For the mnist dataset, Minimal cost achieved is 0.03932 with an accuracy of 89%
- Scikit-Learn's model gives an accuracy of 90%


## Future works

A Convolutional Neural Network would also be a great idea to build and compare with the above three approaches.
