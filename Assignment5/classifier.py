# Casey Owen
# CS131
# Assignment 5, A Radar Trace Classifier
'''
Classifies a set of Radar Traces as either birds or planes based on their velocity and acceleration, using a Naive Recursive Bayesian Classifier.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

class BayesClassifier():
    def __init__(self, t_prob_bird: float, t_prob_plane: float) -> None:
        '''
        Purpose: 
            Classifier Constructor for BayesClassifier class, used for classifying traces as either brids or planes. Loads data, and classifies the traces, using both no feature engineering and acceleration as an additional feature.
        Inputs: 
            -t_prob_bird: The transition probability of birds - the chance that if the object is a bird at timestep t, it is still a bird at timestep t+1.
            -t_prob_plane: The transition probability of planes - the chance that if the object is a plane at timestep t, it is still a plane at timestep t+1. 
        Outputs: none
        '''
        self.loadData()
        self.getAccelLikelihoods()
        self.t_prob_bird = t_prob_bird 
        self.t_prob_plane = t_prob_plane

        self.train_probs_no_fe = np.array(self.classify(self.train_df, 1, False))
        self.train_probs_w_fe = np.array(self.classify(self.train_df, 1, True))
        self.test_probs = np.array(self.classify(self.test_df, 1, True))


    def loadData(self) -> None:
        '''
        Purpose: 
            Loads the traces from files into dataframes, including the known velocity likelihood distributions.
        Inputs: none
        Outputs: none
        '''
        vel_likelihood_df = pd.read_csv('likelihood.txt', sep='\s+', header=None).transpose()
        self.bird_vel_likelihoods = vel_likelihood_df[0].to_numpy()
        self.plane_vel_likelihoods = vel_likelihood_df[1].to_numpy()
        self.vel_likelihood_speeds = np.linspace(0,200,len(self.bird_vel_likelihoods))
        
        self.train_df = pd.read_csv('dataset.txt', sep='\s+', header=None).transpose()
        # 0 for birds 1 for airplanes
        self.actual_training_labels = [0]*10 + [1]*10

        self.test_df = pd.read_csv('testing.txt', sep='\s+', header=None).transpose()



    def getAccels(self, velocity_df: pd.DataFrame, t_interval: int) -> list[list[float]]:
        '''
        Purpose: 
            Calculates the accelerations from a dataframe of velocities, and returns them as a list of lists, one for each column in the dataframe.
        Inputs: 
            -velocity_df: The dataframe of velocities to calculate the accelerations from
            -t_interval: The time interval between velocity recordings, in seconds
        Outputs: 
            -accels: A list of lists of the acceleration values, one for each column in the velocities dataframe.
        '''
        accels = []
        velocity_df = velocity_df
        for column in velocity_df.columns:
            y = velocity_df[column].to_numpy()
            dx = t_interval #1 second
            accel = np.gradient(y)/dx
            accels.append(accel)
        return accels

    def getAccelLikelihoods(self) -> None:
        '''
        Purpose: 
            Gets the likelihoods of the given accelerations by assuming they are normally distributed, and fitting a distribution to the data. This assumption is also checked in plots later. Stores these distributions as class variables to be used later.
        Inputs: none
        Outputs: none 
        '''
        # Assume Normal Distributed Accelerations
        self.bird_accels = self.getAccels(self.train_df.loc[:, :9], 1)
        self.plane_accels = self.getAccels(self.train_df.loc[:, 10:19], 1)
        min_accel = min(np.nanmin(self.bird_accels), np.nanmin(self.plane_accels))
        max_accel = max(np.nanmax(self.bird_accels), np.nanmax(self.plane_accels))

        self.bird_accel_x = np.linspace(min_accel, max_accel, 1000)
        self.bird_accel_likelihoods = norm.pdf(self.bird_accel_x, loc=np.nanmean(self.bird_accels), scale = np.nanstd(self.bird_accels))

        self.plane_accel_x = np.linspace(min_accel, max_accel, 1000)
        self.plane_accel_likelihoods = norm.pdf(self.plane_accel_x, loc=np.nanmean(self.plane_accels), scale = np.nanstd(self.plane_accels))

    def getObservations(self, speed: float, accel: float, feat_eng: bool) -> tuple[float, float]:
        '''
        Purpose: 
            Uses the known velocity and acceleration likelihood distributions to get the likelihoods of birds and planes depending on a single speed and acceleration value. Has the option, in the feat_eng argument, to use or ignore the acceleration likelihood, and simply return the velocity likelihood.
        Inputs: 
            -speed: The speed at which to find the likelihoods
            -accel: The acceleration at which to find the likelihoods
            feat_eng: A boolean argument deciding whether or not to include feature engineering when calculating the liklihoods - that is, whether or not to consider acceleration.
        Outputs: 
            l_bird: The likelihood that the given values are a bird
            l_plane: The likelihood that the given values are a plane
        '''
        if speed < 0 or speed > 200:
            print("Cannot get likelihood for speed outside 0 to 200")
            exit()

        l_bird_vel = np.interp(speed, self.vel_likelihood_speeds, self.bird_vel_likelihoods)
        l_plane_vel = np.interp(speed, self.vel_likelihood_speeds, self.plane_vel_likelihoods)

        if feat_eng:
            l_bird_accel = np.interp(accel, self.bird_accel_x, self.bird_accel_likelihoods)
            l_plane_accel = np.interp(accel, self.plane_accel_x, self.plane_accel_likelihoods)
            # Assume they are close to conditionally independent on true state, and can multiply to get overall likelihood
            l_bird = l_bird_vel * l_bird_accel
            l_plane = l_plane_vel * l_plane_accel
        else:
            l_bird = l_bird_vel
            l_plane = l_plane_vel
        return l_bird, l_plane

    def classify(self, velocity_df: pd.DataFrame, t_interval: int, feat_eng: bool) -> list[tuple[float, float]]:
        '''
        Purpose: 
            Wrapper function for the recursiveEstimation function - calls recursive estimattion on each sample of a velocity dataframe and returns the resulting class probabilities.
        Inputs: 
            -velocity_df: The dataframe of velocities to classify the samples of
            -t_interval: The time interval between velocity samples, in seconds
            feat_eng: A boolean argument deciding whether or not to include feature engineering when calculating the liklihoods - that is, whether or not to consider acceleration.
        Outputs: 
            -result_probs: a list of two-tuples of probability values, representing the normalized probabilities of being a bird or plane
        '''
        result_probs = []
        for column in velocity_df:
            velocities = velocity_df[column]
            accels = np.gradient(velocities)/t_interval
            result_probs.append(self.recursiveEstimation(0.5,0.5, 0, velocities.to_numpy(), accels, feat_eng))
        return result_probs
        
    def recursiveEstimation(self, pbird_t: float, pplane_t: float, t: int, velocities: np.ndarray, accels: np.ndarray, feat_eng: bool) -> tuple[float, float]:
        '''
        Purpose: 
            Naive Recursive Bayesian classifier function. Treats the problem as a hidden markov model where the hidden variable is the true class of the object, and the observations. If feature engineering is selected, a second observation is also used, acceleration, which is assumed to be conditionally independent of the velocity, conditioned on the class.
        Inputs: 
            -pbird_t: The probability of the object being a bird at time t, the previous time step
            -pplane_t: The probability of the object being a plane at time t, the previous time step
            -t: The current timestep
            -velocities: A numpy array of the velocities to use as observations
            -accels: A numpy array of the accelerations to use as observations, if feature engineering is selected
            -feat_eng: A boolean that indicates if feature engineering is selected to be used when classifying
        Outputs: 
            -p_bird_t: The resulting normalized probability that the hidden class is a bird
            -p_plane_t: The resulting normalized probability that the hidden class is a plane
        '''
        if t == len(velocities) - 1:
            return pbird_t, pplane_t
        if np.isnan(velocities[t]) or np.isnan(accels[t]):
            # Skip NaNs, don't update probabilities
            return self.recursiveEstimation(pbird_t, pplane_t, t+1, velocities, accels, feat_eng)
        
        # velocities at time t+1, has index t+1-1 = t
        l_bird, l_plane = self.getObservations(velocities[t], accels[t], feat_eng)
        p_bird_tplus1 = l_bird*((self.t_prob_bird*pbird_t)+((1-self.t_prob_bird)*pplane_t))
        p_plane_tplus1 = l_plane*((self.t_prob_plane*pplane_t)+((1-self.t_prob_plane)*pbird_t))
        
        p_bird_tplus1_norm = p_bird_tplus1/(p_bird_tplus1 + p_plane_tplus1)
        p_plane_tplus1_norm = p_plane_tplus1/(p_bird_tplus1 + p_plane_tplus1)
        
        return self.recursiveEstimation(p_bird_tplus1_norm, p_plane_tplus1_norm, t+1, velocities, accels, feat_eng)

    def plotVelLikelihoods(self, fname: str) -> None:
        '''
        Purpose: 
            Creates a plot of the likelihoods of each class given velocity, and saves it to a file
        Inputs: 
            -fname: The name of the file to save the figure as
        Outputs: none
        '''
        fig, ax = plt.subplots()
        ax.plot(self.vel_likelihood_speeds, self.plane_vel_likelihoods)
        ax.plot(self.vel_likelihood_speeds, self.bird_vel_likelihoods)
        ax.set_title('Likelihoods of Birds and Planes Given Velocity')
        ax.set_xlabel('Velocity (kt)')
        ax.set_ylabel('Likelihood of Class Given Velocity')
        ax.set_ylim(ymin=0)
        ax.legend(['Planes', 'Birds'])
        fig.savefig(fname)


    def plotAccelLikelihoods(self, bird_fname: str, plane_fname: str, combined_fname: str) -> None:
        '''
        Purpose: 
            Creates 3 plots of the likelihoods of each class given acceleration, one for each the assumed distributions plotted against the training data, and one that plots the two distributions against each other for comparison, and saves them to files
        Inputs: 
            -bird_fname: The name of the file to save the figure that plots the bird distribution as
            -plane_fname: The name of the file to save the figure that plots the plane distribution as
            -combined_fname: The name of the file to save the figure that plots the two distributions against each other as
        Outputs: none
        '''
        b_fig, b_ax1 = plt.subplots()
        b_ax1.hist(np.array(self.bird_accels).flatten(), bins='auto')
        b_ax2 = b_ax1.twinx()
        b_ax2.plot(self.bird_accel_x, self.bird_accel_likelihoods, color='orange')
        b_ax2.set_ylim(ymin=0)
        b_ax1.set_title('Bird Acceleration Distribution')
        b_ax1.set_xlabel('Acceleration (kt/s)')
        b_ax1.set_ylabel('Frequency of Acceleration of Birds in Training Data')
        b_ax2.set_ylabel('Likelihood of Being a Bird Given Acceleration')
        b_ax1.legend(['Observed Values Count'], loc='upper left')
        b_ax2.legend(['Assumed Distribution'], loc='upper right')

        p_fig, p_ax1 = plt.subplots()
        p_ax1.hist(np.array(self.plane_accels).flatten(), bins='auto')
        p_ax2 = p_ax1.twinx()
        p_ax2.plot(self.plane_accel_x, self.plane_accel_likelihoods, color='orange')
        p_ax2.set_ylim(ymin=0, ymax=0.7)
        p_ax1.set_title('Plane Acceleration Distribution')
        p_ax1.set_xlabel('Acceleration (kt/s)')
        p_ax1.set_ylabel('Frequency of Acceleration of Planes in Training Data')
        p_ax2.set_ylabel('Likelihood of Being a Plane Given Acceleration')
        p_ax1.legend(['Observed Values Count'], loc='upper left')
        p_ax2.legend(['Assumed Distribution'], loc='upper right')

        c_fig, c_ax = plt.subplots()
        c_ax.plot(self.bird_accel_x, self.bird_accel_likelihoods)
        c_ax.plot(self.plane_accel_x, self.plane_accel_likelihoods)
        c_ax.set_ylim(ymin=0)
        c_ax.set_title('Likelihoods of Birds and Planes Given Acceleration')
        c_ax.set_xlabel('Acceleration (kt/s)')
        c_ax.set_ylabel('Likelihood of Class Given Acceleration')
        c_ax.legend(['Birds', 'Planes'])

        b_fig.savefig(bird_fname)
        p_fig.savefig(plane_fname)
        c_fig.savefig(combined_fname)

    def plotDataExploration(self, bird_data_fname: str, plane_data_fname: str) -> None:
        '''
        Purpose: 
            Creates 2 plots of the provided data, one that plots all 10 known birds of the dataset over time, and one that does the same for planes, and saves them to files. The intent is to identify interesting characteristics of the data to feature engineer.
        Inputs: 
            -bird_data_fname: The name of the file to save the figure that plots the bird data as
            -plane_data_fname: The name of the file to save the figure that plots the plane data as
        Outputs: none
        '''
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        ax1.plot(self.train_df.loc[:, :9])
        ax1.set_title('Velocities of 10 Known Birds')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Velocity (kt)')
        fig1.savefig(bird_data_fname)
        ax2.plot(self.train_df.loc[:, 10:19])
        ax2.set_title('Velocities of 10 Known Planes')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Velocity (kt)')
        fig2.savefig(plane_data_fname)

    def plotTrainResults(self, no_fe_fname: str, w_fe_fname: str) -> None:
        '''
        Purpose: 
            Creates 2 plots of the results of the classification of the training data, one that uses feature engineering and one that does not, and saves them to files. Also prints the actual and calculated labels to the terminal, with the classifier accuracy.
        Inputs: 
            -no_fe_fname: The name of the file to save the figure that plots the results with no feature engineering
            -w_fe_fname: The name of the file to save the figure that plots the results with feature engineering
        Outputs: none
        '''
        # Without Feature Engineering
        x = np.array(range(20))
        fig1, ax1 = plt.subplots()
        ax1.bar(x, self.train_probs_no_fe[:, 1], color='r')
        ax1.plot(x, self.actual_training_labels, marker='*', markersize=15, mfc = 'k', mec = 'k', linestyle = 'None')
        ax1.set_ylabel('Probability of Being a Plane')
        ax1.set_xlabel('Training Trial')
        ax1.legend(['Actual Value', 'Calculated Value'])
        ax1.set_title('Model Evaluation of Training Data before Feature Engineering')
        ax1.set_xticks(range(20))
        fig1.savefig(no_fe_fname)

        # With Feature Engineering
        fig2, ax2 = plt.subplots()
        ax2.bar(x, self.train_probs_w_fe[:, 1], color='r')
        ax2.plot(x, self.actual_training_labels, marker='*', markersize=15, mfc = 'k', mec = 'k', linestyle = 'None')
        ax2.set_ylabel('Probability of Being a Plane')
        ax2.set_xlabel('Training Trial')
        ax2.legend(['Actual Value', 'Calculated Value'])
        ax2.set_title('Model Evaluation of Training Data after Feature Engineering')
        ax2.set_xticks(range(20))
        fig2.savefig(w_fe_fname)

        # Print Labels to terminal
        print('Train Data Results')
        print(f'{self.convertToab(self.actual_training_labels)} - Actual Training Labels')
        calc_train_labels_no_fe = (self.train_probs_no_fe[:,1] > 0.5)*1
        train_acc_no_fe = sum(calc_train_labels_no_fe == self.actual_training_labels)/len(self.actual_training_labels)*100
        print(f'{self.convertToab(calc_train_labels_no_fe)} {train_acc_no_fe}% Accuracy - Calculated Training Labels, without feature engineering')
        calc_train_labels_w_fe = (self.train_probs_w_fe[:,1] > 0.5)*1
        train_acc_w_fe = sum(calc_train_labels_w_fe == self.actual_training_labels)/len(self.actual_training_labels)*100
        print(f'{self.convertToab(calc_train_labels_w_fe)} {train_acc_w_fe}% Accuracy - Calculated Training Labels, with feature engineering')

    def plotTestResults(self, fname: str) -> None:
        '''
        Purpose: 
            Creates a plots of the results of the classification of the test data, using feature engineering, and saves it to a file. Also prints the actual and calculated labels to the terminal, with the classifier accuracy.
        Inputs: 
            -fname: The name of the file to save the figure to
        Outputs: none
        '''
        x = np.array(range(10))
        fig, ax = plt.subplots()
        ax.bar(x, self.test_probs[:, 1], color='r')
        plt.ylabel('Probability of Being a Plane')
        plt.xlabel('Test Trial')
        ax.set_title('Model Evaluation of Test Data')
        ax.set_xticks(range(10))
        fig.savefig(fname)

        # Print Labels to Terminal
        print('Test Data Results')
        test_labels = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 0])
        print(f'{self.convertToab(test_labels)} - Actual Testing Labels')
        calc_test_labels = (self.test_probs[:,1] > 0.5)*1
        test_acc = sum(calc_test_labels == test_labels)/len(test_labels)*100
        print(f'{self.convertToab(calc_test_labels)} {test_acc}% Accuracy - Calculated Testing Labels')

    def convertToab(self, arr) -> list[str]:
        '''
        Purpose: 
            Converts an array of 1's and 0's to 'a's and 'b's respectively
        Inputs: 
            -arr: The array to convert
        Outputs: 
            -converted: The converted array
        '''
        return list(map(lambda x: 'a' if x==1 else 'b', arr))
