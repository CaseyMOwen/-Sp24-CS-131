# Casey Owen
# CS131
# Assignment 6, Gardens of Heaven
'''
Implementation of a layer of a simple trainable, feed-forward, fully connected neural network
'''
import numpy as np
from neuron import Neuron

class Layer():
    def __init__(self, num_neurons: int, num_inputs: int, activation_fcn: str, learning_rate: float) -> None:
        '''
        Purpose: 
            Constructor for Layer class, used as a layer of a trainable, feed-forward, fully connected neural network
        Inputs: 
            -num_neurons: The number of neurons that the layer has
            -num_inputs: The number of parent connections each neuron will have
            -activation_fcn: The activation function that every neuron will use, can be either 'sigmoid' or 'tanh'
            -learning_rate: Adjustable hyperparameter that affects overfitting
        Outputs: none
        '''
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.activation_fcn = activation_fcn
        self.neurons = np.array([Neuron(num_inputs, activation_fcn, learning_rate) for _ in range(num_neurons)])

    def set_inputs(self, input_array: list[float]) -> None:
        '''
        Purpose: 
            Sets the inputs of the layer by setting those inputs to be the inputs for each neuron
        Inputs: 
            -input_array: The inputs to set for the layer
        Outputs: none
        '''
        if len(input_array) != self.num_inputs:
            raise ValueError("Number of inputs provided to the layer must match the originally specified number of inputs")
        for neuron in self.neurons:
            neuron.set_inputs(input_array)
    
    def connect(self, parent_layer: 'Layer') -> None:
        '''
        Purpose: 
            Connnects a layer to its parent layer by setting the parent layer to be the parents for each neuron in the current layer
        Inputs: 
            -parent_layer: The parent layer to set for the current layer
        Outputs: none
        '''
        for neuron in self.neurons:
            neuron.set_parents(parent_layer.neurons)
    
    def forward_propogate(self):
        '''
        Purpose: 
            Forward propogates values through the layer for each neuron. This calls on each neuron to set its parents ouptput as its current input
        Inputs: none
        Outputs: none
        '''
        for neuron in self.neurons:
            neuron.forwardpropogate()

    def set_desired_outputs(self, desired_outputs: list[float]) -> None:
        '''
        Purpose: 
            Sets the desired output of a layer - should only be called on the output layer, otherwise these desired outputs are automatically calculated, based on downstream weights and desired outputs
        Inputs: 
            -desired_outputs: Array of outputs to set for the layer, that matches the layer size
        Outputs: none
        '''
        if len(desired_outputs) != self.num_neurons:
            raise ValueError("Number of desired outputs provided to the layer must match the originally specified number of neurons")
        for i, neuron in enumerate(self.neurons):
            neuron.set_desired_output(desired_outputs[i])

    def backpropogate(self):
        '''
        Purpose: 
            Backpropogates the desired outputs of the layer and updates the layers weights, and updates the parents signal error. Should only be called once the next sequential layer has already been backpropogated, or this layers desired outputs have been set (if it is the output layer)
        Inputs: none
        Outputs: none
        '''
        for neuron in self.neurons:
            neuron.backpropogate()
    
    @property
    def output_array(self) -> list[float]:
        '''
        The list of outputs of each neuron in the layer
        '''
        return [neuron.output for neuron in self.neurons]
    