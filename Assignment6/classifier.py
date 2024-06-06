# Casey Owen
# CS131
# Assignment 6, Gardens of Heaven
'''
Trains a neural network model based off of a given dataframe of features and a class, where the class is the last column. Has functionality for performing grid search hyperparemeter tuning to choose the best model, then classify future inputs with that model
'''
import numpy as np
import pandas as pd
from neuralnetwork import NeuralNetwork
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

class NNClassifier():
    def __init__(self, data_df:pd.DataFrame) -> None:
        '''
        Purpose: 
            Classifier Constructor for NNClassifier class, used for performing grid search hyperparemeter tuning to choose the best model, then classify future inputs with that model.
        Inputs: 
            -data_df: the data on which to base the model. All columns must be features, except for the rightmost column, which is the labelled class.
        Outputs: none
        '''
        self.data_df = data_df.sample(frac=1)
        self.num_features = self.data_df.shape[1] - 1
        self.classes = set(self.data_df.iloc[:, -1])
        self.num_classes = len(self.classes)
        # Drop class column and add one hot encoding df
        one_hot_df = 0.0+pd.get_dummies(self.data_df.iloc[:,-1])
        self.data_df = self.data_df.iloc[:,:-1].join(one_hot_df)
        self.data_df = self.data_preprocessing(self.data_df)

    def data_preprocessing(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        Purpose: 
            Normalizes a given dataframe with sklearns MinMaxScaler
        Inputs: 
            -df: The dataframe to normalize
        Outputs: 
            -df_scaled: The normalized dataframe
        '''
        self.scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns)
        return df_scaled

    def create_hyperparemeter_df(self, hyperparameters: dict) -> pd.DataFrame:
        '''
        Purpose: 
            Creates a hyperparameter dataframe, that includes all possible combinations (the cross product) of the given hyperparameters
        Inputs: 
            -hyperparameters: A dict of all possible values of each hyperparameter to take the cross product of. Must include keys "learning_rate", "hidden_layer_size", and "activation_fcn", where the values are lists of all possible values that parameter should take on
        Outputs: 
            -cross_product_df: A dataframe where each row is a possible combo of hyperparameters
        '''
        cross_product_df = None
        for key in hyperparameters:
            hyp_df = pd.DataFrame({key:hyperparameters[key]})
            if cross_product_df is None:
                cross_product_df = hyp_df
            else:
                cross_product_df = pd.merge(cross_product_df, hyp_df, how='cross')
        return cross_product_df
    
    def split_data(self, train_frac: pd.DataFrame, validation_frac: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        Purpose: 
            Splits the provided data to the class into three datasets - training data, validation data, and testing data based on the fractions provided. Evenly splits by class, so that each created subset has matching proportions of each class
        Inputs: 
            -train_frac: The fraction of provided data that is split into training data
            -valid_frac: The fraction of provided data that is split into validation data. All remaining data is split into the test set
        Outputs: 
            -train_df: A dataframe of the created training data
            -validation_df: A dataframe of the created validation data
            -test_df: A dataframe of the created testing data
        '''
        data_by_class = []
        for i in range(len(self.classes)):
            class_column_num = -i - 1
            segmented_df = self.data_df[self.data_df.iloc[:, class_column_num] == 1]
            data_by_class.append(segmented_df)

        for i, class_df in enumerate(data_by_class):
            first_split_loc = int(train_frac*class_df.shape[0])
            second_split_loc = int((train_frac+validation_frac)*class_df.shape[0])
            class_train_df = class_df.iloc[:first_split_loc]
            class_validation_df = class_df.iloc[first_split_loc:second_split_loc]
            class_test_df = class_df.iloc[second_split_loc:]
            if i == 0:
                train_df = class_train_df
                validation_df = class_validation_df
                test_df = class_test_df
            else:
                train_df = pd.concat([train_df, class_train_df])
                validation_df = pd.concat([validation_df, class_validation_df])
                test_df = pd.concat([test_df, class_test_df])
        return train_df, validation_df, test_df


    def train_model(self, model:NeuralNetwork, train_df:pd.DataFrame, validation_df:pd.DataFrame, max_iter: int, early_stop_iters: int) -> tuple[int, float, float, list[float], list[float]]:
        '''
        Purpose: 
            Trains the given model on training and validation dataframes over several iterations, with parameters for a maximum number of iterations to stop after, and the ability to stop early if the validation error is plateuing or rising. Each iteration the model trains over the entire training set, updating weights at each datapoint, then accuracy is measured at the end of each iteration
        Inputs: 
            -model: The model to train
            -train_df: The dataframe of training data
            -validation_df: The dataframe of validation data
            -max_iter: The maximum number of iterations for the data may train for
            -early_stop_iters: If during training, the validation error rises or stays the same for this many consecutive iterations, it is determined that no more progress is being made, and training is stopped early 
        Outputs: 
            -i: The number of iterations the model took to train
            -train_loss: The final training loss of the model
            -valid_loss: The final validation loss of the model
            -train_err_by_iter: A list of the training errors by iteration
            -valid_err_by_iter: A list of the validation errors by iteration
        '''
        loss = np.inf
        i = 0
        valid_err_by_iter = []
        train_err_by_iter = []
        loss_by_iter = []
        validerr_rising_iters = 0
        while i < max_iter and validerr_rising_iters < early_stop_iters:
            train_df = train_df.sample(frac=1)
            model.train(train_df.iloc[:, :self.num_features], train_df.iloc[:, self.num_features:])
            train_acc, train_loss = self.get_accuracy(model, train_df) 
            valid_acc, valid_loss = self.get_accuracy(model, validation_df) 
            train_err = 1 - train_acc
            valid_err = 1 - valid_acc
            if valid_err_by_iter != [] and valid_err >= valid_err_by_iter[-1]:
                validerr_rising_iters += 1
            else:
                validerr_rising_iters = 0 
            valid_err_by_iter.append(valid_err)
            train_err_by_iter.append(train_err)
            loss_by_iter.append(loss)
            i += 1
        return i, train_loss, valid_loss, train_err_by_iter, valid_err_by_iter 

    def get_accuracy(self, model:NeuralNetwork, df:pd.DataFrame) -> tuple[float, float]:
        '''
        Purpose: 
            Evaluates the accuracy and loss of a given model on a given set of data
        Inputs: 
            -model: The model to evaluate
            -df: The dataframe of data to evaluate it on
        Outputs: 
            -accuracy: The percent of correct classifications the model made
            -loss: The Mean Square Error (MSE) of the models predictions
        '''
        num_correct = 0
        loss = 0
        for i, row in df.iterrows():
            inputs = row[:self.num_features].to_list()
            outputs = row[self.num_features:].to_list()
            pred_outputs = model.predict(inputs)
            if np.argmax(outputs) == np.argmax(pred_outputs):
                num_correct += 1
            point_loss = 0
            for i, output in enumerate(outputs):
                point_loss += (outputs[i] - pred_outputs[i])**2
            loss += point_loss/len(outputs)
        accuracy = num_correct/df.shape[0]
        return accuracy, loss/df.shape[0]



    def choose_best_model(self, hyperparameters: dict, max_iter: int, early_stop_iters: int) -> tuple[NeuralNetwork, pd.DataFrame, float, float, float, list[float], list[float], float]:
        '''
        Purpose: 
            Performs grid search over a set of hyperparameters, training a model at each possible combination, and return the best model, along with a dataframe showing the results of the grid search and stats on the best model, chosen by lowest MSE. Uses a 60/20/20 train/validation/test split.
        Inputs: 
            -hyperparameters: A dict of all possible values of each hyperparameter to take the cross product of. Must include keys "learning_rate", "hidden_layer_size", and "activation_fcn", where the values are lists of all possible values that parameter should take on
            -max_iter: The maximum number of iterations that each model may train for
            -early_stop_iters: If during training, the validation error rises or stays the same for this many consecutive iterations, it is determined that no more progress is being made, and training is stopped early 
        Outputs: 
            -best_model: The resulting best model
            -results_df: A dataframe showing the results of the grid search
            -best_train_loss: The training loss of the best model
            -best_valid_loss: The validation loss of the best model
            -best_test_loss: The testing loss of the best model
            -best_train_err_by_iter: A list of the training errors of the best model by iteration 
            -best_valid_err_by_iter: A list of the validation errors of the best model by iteration
            -best_test_accuracy: The test accuracy of the best model
        '''
        hyperparemeter_df = self.create_hyperparemeter_df(hyperparameters)
        num_models = hyperparemeter_df.shape[0]
        train_df, validation_df, test_df = self.split_data(0.6, 0.2)
        best_model = None
        best_train_loss = None
        best_valid_loss = None
        best_valid_err_by_iter = None
        best_train_err_by_iter = None
        results = []
        for i, row in hyperparemeter_df.iterrows():
            print(f"Training model {i+1}/{num_models}")
            model = NeuralNetwork(self.num_features, row['hidden_layer_size'], self.num_classes, row['activation_fcn'], row['learning_rate'])
            iters, train_loss, valid_loss, train_err_by_iter, valid_err_by_iter = self.train_model(model, train_df, validation_df, max_iter, early_stop_iters)
            accuracy = 1 - valid_err_by_iter[-1]
            results.append([iters, accuracy, valid_loss])
            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_train_loss = train_loss
                best_valid_loss = valid_loss
                best_model = deepcopy(model)
                best_valid_err_by_iter = valid_err_by_iter
                best_train_err_by_iter = train_err_by_iter
        results_df = pd.DataFrame(results, columns=['Iterations', 'Validation Accuracy', 'Validation MSE'])
        results_df = pd.concat([hyperparemeter_df, results_df], axis=1).sort_values(by=['Validation MSE'], ascending=True)
        best_test_accuracy, best_test_loss = self.get_accuracy(best_model, test_df)
        return best_model, results_df, best_train_loss, best_valid_loss, best_test_loss, best_train_err_by_iter, best_valid_err_by_iter, best_test_accuracy
    
    def classify(self, model:NeuralNetwork, inputs:pd.DataFrame) -> tuple[str, float]:
        '''
        Purpose: 
            Classifies a given set of inputs using a given model
        Inputs: 
            -model: The model to use to classify
            -inputs: A dataframe of inputs to classify, must match the format of the original dataframe passed into the constructor
        Outputs: 
            -class_name: The name of the calculated class
            -certainty: The score assigned to the output neuron of the calculated class
        '''
        scaled_inputs = self.scaler.transform(inputs)
        pred_outputs = model.predict(scaled_inputs[0, :self.num_features])
        pred_class = np.argmax(pred_outputs)
        class_name = self.data_df.columns[self.num_features + pred_class]
        certainty = pred_outputs[pred_class]
        return class_name, certainty
