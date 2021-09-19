# Neural Network from Scratch
Building a Neural Network for UNIBE ACAS DS 2021

#### Contributers:
- Maria Mumtaz, maria.mumtaz@students.unibe.ch
- Davi Bicudo, davi.bicudo@students.unibe.ch

## Description

In this repository, a fully connected 5 layers Neural Network is build. Then Neural Network is then used to recognize Images in mnist dataset(hand-written digits)
and cvnc dataset(Cat vs. No Cat)

## Technologies used

First a neural network was built using pytorch. The implementation can be found in *LibNuralNetwork.ipynb*. Then using numpy, a similar neural Network is build from scratch. This implementation can be found in NeuralNetExampleFinal.ipynb. It trains the model and uses gradient decent to compute the costs.
The self made model is then compared to scikit-learn's Multi-Layer Perceptron to evaluate the quality of our Model

## Structure

The directory structure is as below

**src**: All python source Files
**model**: Saved trained models
**data**: mnist and cvnc test/train sets 
**images**: Test images 
**\<root\>**: Main python file and all scratch notebooks

## Source directiry

- **activation** : has activation all activation functions and their derivatives (e.g sigmoid, sigmoid_prime)
- **loss** : computes the losses
- **config**: sets up seed, number of classes, number of epochs, learning rate
- **util**: has utility functions like dump_model, load_model
- **data_loader**: loads the data and returns train and test sets
- **nn_unit**: abstract class for any nerual network unit
- **fc_unit**: implementation of nn_unit (has forward and backward propagation functions, and calculates weights)
- **nn_layer**: abstract class for neural network layer
- **fc_layer**: Self implementation of a fully coonected layer
- **nn_network**: adds layers, trains model, computes costs, predicts and evaluate


## Results

- For the cvnc dataset, Minimal cost achieved is 0.00073 with an accuracy of 76%
  ![alt text](https://raw.githubusercontent.com/MariaMumtaz/NeuralNetProj/master/images/cvnc.png)
  
- For the mnist dataset, Minimal cost achieved is 0.03932 with an accuracy of 89%
- ![alt text](https://raw.githubusercontent.com/MariaMumtaz/NeuralNetProj/master/images/minst.png)
  
- Scikit-Learn's MLP model gives an accuracy of 90%


## Future works

A Convolutional Neural Network would also be a great idea to build and compare with the above three approaches.
