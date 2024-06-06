# Casey Owen
# CS131
# Assignment 6, Gardens of Heaven
'''
Implementation of the Gardens of Heaven Assignment, main.py. Runs the main
program, accepts user input, and prints output to terminal.
'''

from classifier import NNClassifier
import pandas as pd
import matplotlib.pyplot as plt

def main():
    '''
    Purpose: 
        Sets up the program, trains the models, and calls the accept_queries function until the program is complete
    Inputs: none
    Outputs: none
    '''
    print("\nWelcome to the Gardens of Heaven!")
    print("\nThis tool allows you to train a neural network to classify 3 types of iris plants based on four important features - Sepal Length, Sepal Width, Petal Length, and Petal Width.")
    print("The possible iris plant types are Iris Setosa, Iris Versicolour, and Iris Virginica.")
    input("\nPress [Enter] when you are ready to begin training the model.\nThis program will search through 32 possible combinations of hyperparameters in order to find the combination that performs best, so it may take a minute.\n")

    data_df = pd.read_csv('ANN - Iris data.txt', names=["sepal length", "sepal width", "petal length", "petal width", "class"])
    nnc = NNClassifier(data_df)
    hyperparameters = {
            'learning_rate': [0.1, 0.5, 1, 5],
            'hidden_layer_size' : [[3], [5], [3,3], [3,1]],
            'activation_fcn': ['tanh', 'sigmoid']
        }
    
    # An Alternative hyperparameter set to use to speed up the model training - this model tends to perform well
    
    # hyperparameters = {
    #         'learning_rate': [5],
    #         'hidden_layer_size' : [[5]],
    #         'activation_fcn': ['sigmoid']
    #     }
    
    model, results_df, train_loss, valid_loss, test_loss, train_err_by_iter, valid_err_by_iter, test_accuracy = nnc.choose_best_model(hyperparameters, 100, 25)
    print("\nThe best model has been found!")
    print("\nThe following table shows details of the results of hyperparameter tuning:")
    print(results_df)

    print(f"\nThe best model uses the following hyperparameters:")
    print(f"Hidden Layers: {model.hidden_layer_sizes}")
    print(f"Activation Function: {model.activation_fcn}")
    print(f"Learning Rate: {model.learning_rate}")

    print("\nThe model also has the following statistics on the data provided:")

    print(f"Training Accuracy: {1 - train_err_by_iter[-1]}")
    print(f"Validation Accuracy: {1 - valid_err_by_iter[-1]}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"\nTraining Mean Square Error (MSE): {train_loss}")
    print(f"Validation Mean Square Error (MSE): {valid_loss}")
    print(f"Test Mean Square Error (MSE): {test_loss}")

    plt.subplots()
    plt.plot(range(len(train_err_by_iter)), train_err_by_iter)
    plt.plot(range(len(valid_err_by_iter)), valid_err_by_iter)
    plt.title('Training Progress of Best Model')
    plt.xlabel('Iteration')
    plt.ylabel('Percent of Inaccurate Predictions')
    plt.legend(['Training Error', 'Validation Error'])
    plt.savefig('best_model_training_graph.png')

    print("\n A graph showing the training progress of the best model has just been saved to file as 'best_model_training_graph.png'")

    print("\nNow that the model has been trained and selected, you may input a query in the form of the 4 plant features, and it will be classified. At any time, you may enter 'quit' to quit.")
    done = False
    while not done:
        done = accept_queries(nnc, model)
        cmd = input("\nWould you like to enter another plant? y/n:\n")
        if cmd == 'y':
            done = False
        elif cmd == 'n':
            done = True
            print('Goodbye!')

def accept_queries(nnc, model) -> bool:
    '''
    Purpose: 
        Runs the primary program logic, accepts user input and handles commands
    Inputs: none
    Outputs: 
        -done: a boolean indicating if the program is done running
    '''
    try:
        query = []
        features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        for feature in features:
            badinput = True
            while badinput:
                cmd = input("Please enter the " + feature + " of your plant: ")
                if cmd == "quit":
                    return True
                try:
                    num = float(cmd)
                    badinput = False
                    query.append(num)
                except:
                    print("\nError: You must enter a value that can be parsed as a float\n")
        toclassify = pd.DataFrame({'sepal length' : [query[0]],
                               'sepal width': [query[1]],
                               'petal length': [query[2]],
                               'petal width': [query[3]],
                               'Iris-setosa': [0],
                               'Iris-versicolor': [0],
                               'Iris-virginica': [0]})
        classname, certainty = nnc.classify(model, toclassify)
        print(f'\nYour plants class: {classname}')
        print(f'Certainty: {certainty:.1%}')
    except KeyboardInterrupt:
        return True

if __name__ == "__main__":
    main()