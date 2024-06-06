# Sp24-CS-131

Work done for Tufts University's CS131 Aritificial Intelligence class in Spring 2024.

As part of this graduate-level class I learned a wide variety of topics relevant to AI and implemented many of them in assignments with python. Some of the topics covered in this class:
- Behavior Trees
- Uninformed Search
- Informed Search (Heuristics, A*)
- Genetic Algorithms
- Constraint Satisfaction Problems
- Propositional and First Order Logic
- Probability Theory and Bayesian Networks
- Markov Models
- Machine Learning paradigms
- Decision Trees
- Neural Networks
- Clustering

The assignments listed below, with the source code in the repository, represent the work I completed during the semester, using python. All but Assignment 1 was completed from scratch. 

All assignments can be run from the code provided, with instructions and more details in the corresponding README's. In general, they can be run by running main.py along with the installation of python (I used 3.10.12) and some basic packages.

## Assignment 1

From starter code, independently created a **decision tree** representing the logic of a robot vacuum cleaner. Uses an object-oriented programming (OOP) structure in python.

To run, **run Assignment1/main.py**

Libraries required:
- random
## Assignment 2

Independently created an algorithmic solution to "The Pancake Problem" where 10 differently sized pancakes are in a stack, and are desired to be put in order. The pancakes can be flipped by a spatula that can be inserted at any point in the stack, and flips all pancakes above it. 

My python solution defines the problem as a **search**, uses a custom **heuristic** function, and solves this using either **A\*** or **Uniform Cost Search (UCS)** as the user requests.

To run, **run Assignment2/main.py**

Libraries required:
- random
- copy
- heapq
## Assignment 3

Independently created an algorithmic solution to "The Knapsack Problem" where a set number of boxes (with corresponding weights and values) need to be fit into a backpack. The backpack has a maximum weight capacity, and the goal is to maximize the value of the boxes added to the backpack.

My python solution solves this with a **genetic algorithm** where which boxes are in the backpack is represented by a genome - a string of 1's and 0's. Then, the genetic algorithm selects the best genome, which is the set of boxes that maximizes the total box value.

To run, **run Assignment3/main.py**

Libraries required:
- random
- copy
- numpy
- matplotlib
## Assignment 4

Independently created an solver for 9x9 Sudoku puzzles from scratch. 

My solution frames the problem as a **Constraint Satisfaction Problem** - in principle, possible values are guessed and checked against constraints. However, this method by itself, would be intractable for non-trivial puzzles. I included several optimizations, including **forward checking**, **variable ordering**, and **value ordering** in order to make the solution tractable. After these improvements, even very hard puzzles can be solved in fractions of a second.

To run, **run Assignment4/main.py**

Libraries required:
- csv
- copy
- numpy
- time
- math
## Assignment 5

Independently created a **Naive Bayes Classifier** from scratch. The assignment provides datasets about the known **likelihood distributions** of velocities birds and planes, as tracked by radar at an airport, and asks to classify future objects based on their velocity profile alone. 

The assignment also provides datasets of velocity profiles that are known to be bird or planes so that new, custom features can be added to the model (**feature engineering**) - in this case, I used acceleration, since there was also a significant amount of class info encoded in the accelerations of the objects.

The resulting model, after adding the acceleration feature, correctly classified 9/10 objects in the test data. The results of this model are best explored through the produced figures in the "figures" folder.

To run, **run Assignment5/main.py**

Libraries required:
- pathlib
- matplotlib
- numpy
- pandas
- scipy

## Assignment 6

Independently created an **Artificial Neural Network** from scratch to classify Iris plants. The assignment provides data from the well-known Fisher's Iris database, which includes 50 instances each of 3 classes of Iris plant, along with 4 attributes for each instance.

The neural network was created using an object-oriented programming (OOP) structure of neurons, layers, and network to implement both forward- and back-propagation.

The models are trained using stochastic gradient descent, optimizing for mean square error. To select the best hyper-parameters (number/sizes of hidden layers, activation function, learning rate), a grid search is performed over 32 possible models, and the best is selected to allow users to input properties of a given Iris plant, and receive the class in return.

The best model resulting from hyperparameter tuning tends to perform very well, always over 90% accuracy, often getting 29/30 or 30/30 classes correct on test data.

To run, **run Assignment6/main.py**

Libraries required:
- sklearn (for MinMaxScaler only)
- matplotlib
- numpy
- pandas
- copy
