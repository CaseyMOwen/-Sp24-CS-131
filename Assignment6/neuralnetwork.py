# Casey Owen
# CS131
# Assignment 6, Gardens of Heaven
'''
Implementation of a simple trainable, feed-forward, fully connected neural network with a generalizable number and sizes of hidden layers
'''
from layer import Layer
import pandas as pd

class NeuralNetwork():
    def __init__(self, num_inputs: int, hidden_layer_sizes: list[int], num_outputs: int, activation_fcn: str, learning_rate: float) -> None:
        '''
        Purpose: 
            Constructor for NeuralNetwork class, used as a trainable, feed-forward, fully connected neural network with a generalizable number and sizes of hidden layers
        Inputs: 
            -num_inputs: The number of inputs that the neural network may take (the number of input neurons)
            -hidden_layer_sizes: A list of sizes for all hidden layers desired in the network. For none, pass an empty list
            -num_outputs: The number of outputs that the neural network should provide (the number of output neurons)
            -activation_fcn: The activation function that every neuron will use, can be either 'sigmoid' or 'tanh'
            -learning_rate: Adjustable hyperparameter that affects overfitting
        Outputs: none
        '''
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_inputs = num_inputs
        self.activation_fcn = activation_fcn
        self.learning_rate = learning_rate

        # Input Layer
        input_layer = Layer(self.num_inputs, self.num_inputs, activation_fcn, learning_rate)
        self.layers = [input_layer]
        prev_layer = input_layer

        # Hidden Layers
        for hidden_layer_size in self.hidden_layer_sizes:
            new_layer = Layer(hidden_layer_size, prev_layer.num_neurons, activation_fcn, learning_rate)
            new_layer.connect(prev_layer)
            self.layers.append(new_layer)
            prev_layer = new_layer
        
        # Output Layer
        output_layer = Layer(num_outputs, prev_layer.num_neurons, activation_fcn, learning_rate)
        output_layer.connect(prev_layer)
        self.layers.append(output_layer)

    def forward_propogate(self, input_array: list[float]) -> list[float]:
        '''
        Purpose: 
            Propogates a set of inputs forward through the neural network, such that all layers have defined outputs afterwards, including the final layer, which is returned as the result
        Inputs: 
            -input_array: List of numbers of length matching the number of input neurons, where each element represents the input to the neuron at that index
        Outputs: 
            -output_array: List of numbers of length matching the number of output neurons, where each element represents the output of the neuron at that index
        '''
        if len(input_array) != self.num_inputs:
            raise ValueError("Number of inputs provided to the network must match the originally specified number of inputs")
        for i, layer in enumerate(self.layers):
            if i == 0:
                self.layers[0].set_inputs(input_array)
            else:
                self.layers[i].forward_propogate()
        return self.layers[-1].output_array
    
    def backpropogate(self, desired_outputs: list[float]) -> None:
        '''
        Purpose: 
            Propogates a set of desired outputs backward through the neural network, updating the weights at each layer. Must be called after forward propogation of the set of inputs that creates those desired outputs
        Inputs: 
            -desired_outputs: List of numbers of length matching the number of output neurons, where each element represents the desired output of the neuron at that index
        Outputs: none
        '''
        self.layers[-1].set_desired_outputs(desired_outputs)
        for layer in reversed(self.layers):
            layer.backpropogate()
    
    def train(self, input_df: pd.DataFrame, outputs_df: pd.DataFrame) -> None:
        '''
        Purpose: 
            Trains the neural network on a given dataframe of inputs and outputs by repeatedly calling forward_propogate and backpropogate on each row.
        Inputs: 
            -input_df: A dataframe of input values, where each row is a valid input_array to the network
            -output_df: A dataframe of output values to train on, which the same number of rows as the input_df
        Outputs: none
        '''
        if input_df.shape[0] != outputs_df.shape[0]:
            raise ValueError("Input dataframe must have same number of rows as output dataframe")
        for i in range(input_df.shape[0]):
            self.forward_propogate(input_df.iloc[i, :].to_list())
            self.backpropogate(outputs_df.iloc[i, :].to_list())
    
    def predict(self, input_array: list[float]) -> list[float]:
        '''
        Purpose: 
            Alias for the forward_propogate function, representing what it may be used for. Given a set of inputs, makes a predition on the values of the output neurons
        Inputs: 
            -input_array: List of numbers of length matching the number of input neurons, where each element represents the input to the neuron at that index
        Outputs: 
            -output_array: List of numbers of length matching the number of output neurons, where each element represents the output of the neuron at that index
        '''
        return self.forward_propogate(input_array)
