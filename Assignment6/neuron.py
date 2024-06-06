# Casey Owen
# CS131
# Assignment 6, Gardens of Heaven
'''
Implementation of a neuron of a simple trainable, feed-forward, fully connected neural network
'''
import numpy as np

class Neuron():
    def __init__(self, num_connections: int, activation_fcn: str, learning_rate: float) -> None:
        '''
        Purpose: 
            Constructor for Neuron class, used as a neuron of a trainable, feed-forward, fully connected neural network
        Inputs: 
            -num_connections: The number of parent connections the neuron will have
            -activation_fcn: The activation function that the neuron will use, can be either 'sigmoid' or 'tanh'
            -learning_rate: Adjustable hyperparameter that affects overfitting
        Outputs: none
        '''
        self.num_connections = num_connections
        self.activation_fcn = activation_fcn
        self.learning_rate = learning_rate
        # Also has bias weight
        num_inputs = self.num_connections + 1
        self.weights = np.random.uniform(low=-1/num_inputs, high=1/num_inputs, size = num_inputs)
        self.inputs = None
        # Array of nuerons matching size of num_connections - if input layer, is None
        self.parents = None
        self.signal_error = 0

    def set_inputs(self, input_array: list[float]):
        '''
        Purpose: 
            Sets the input values for the neuron, not including the bias weight. Only call directly on the input layer, otherwise use forwardpropogate
        Inputs: 
            -input_array: List of input values with length matching the originally specified number of connections
        Outputs: none
        '''
        if len(input_array) != self.num_connections:
            raise ValueError("Number of inputs provided to the neuron must match the originally specified number of connections")
        # Adding the bias input of 1 to the end
        self.inputs = np.append(input_array, [1])

    def forwardpropogate(self):
        '''
        Purpose: 
            Sets the input values for the neuron to be the outputs of its parents. Only use after parents have been set
        Inputs: none
        Outputs: none
        '''
        self.set_inputs([parent.output for parent in self.parents])

    # Only call for for output layer
    def set_desired_output(self, desired_output: list[float]):
        '''
        Purpose: 
            Sets the desired output values for the neuron, and updates the neurons signal error, used for backpropogation. Only use on neurons in the output layer, otherwise signal error is calculated automatically as a part of backpropogation
        Inputs: none
        Outputs: none
        '''
        self.signal_error = self.output_deriv*(desired_output - self.output)
        
    @property
    def output_deriv(self) -> float:
        '''
        The derivative of the output of the neuron, the formula of which depends on the activation function
        '''
        if self.activation_fcn == 'sigmoid':
            return self.output*(1 - self.output)
        elif self.activation_fcn == 'tanh':
            return 1 - (self.output**2)

    def backpropogate(self):
        '''
        Purpose: 
            Adds contribution of signal error to parents, and updates current neurons weights, then resets signal error back to 0. Since this only adds this neurons contribution of signal error to its parent, this function should always be called on every neuron in its layer
        Inputs: none
        Outputs: none
        '''
        if self.parents is not None:
            for i, parent in enumerate(self.parents):
                # Add individual component of signal error to parents (since it is initialized at 0, will result in full sum so long as this is called on every neuron in the layer)
                parent.signal_error += parent.output_deriv*self.weights[i]*self.signal_error
                self.weights[i] += self.learning_rate*parent.output*self.signal_error
            # Bias weight
            self.weights[-1] += self.learning_rate*self.signal_error
        else: # Input Layer
            for i, input in enumerate(self.inputs):
                self.weights[i] += self.learning_rate*input*self.signal_error
        # Reset signal error for next datapoint to train on, now that weights are updated
        self.signal_error = 0

    @property
    def potential(self) -> float:
        '''
        The potential of the neuron, which is the dot product of the weights and inputs.
        '''
        if self.inputs is None:
            raise ValueError("Cannot get potential until inputs have been provided")
        return np.dot(self.inputs, self.weights)

    @property
    def output(self) -> float:
        '''
        The output of the neuron as a function of the potential, the formula of which depends on the activation function
        '''
        if self.activation_fcn == 'sigmoid':
            return 1/(1 + np.exp(-self.potential))
        elif self.activation_fcn == 'tanh':
            return np.tanh(self.potential)
        
    def set_parents(self, parents: list['Neuron']) -> None:
        '''
        Purpose: 
            Sets the parents of the neuron.
        Inputs: 
            -parents: The list of neurons to set as the parents of the current neuron
        Outputs: none
        '''
        self.parents = parents