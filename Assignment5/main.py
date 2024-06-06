# Casey Owen
# CS131
# Assignment 5, A Radar Trace Classifier
'''
Implementation of the Naive Bayesian classification Assignment, main.py. Classifies the results, saves the figures to files, and prints output to terminal.
'''

from classifier import *
from pathlib import Path

def main() -> None:
    '''
    Purpose: 
        Sets up the program, and prints the resulting figures to convenient locations within a directory structure.
    Inputs: none
    Outputs: none
    '''
    b = BayesClassifier(0.9,0.9)
    Path("figures/").mkdir(parents=True, exist_ok=True)
    Path("figures/data_exploration/").mkdir(parents=True, exist_ok=True)
    Path("figures/likelihoods/").mkdir(parents=True, exist_ok=True)
    Path("figures/results/").mkdir(parents=True, exist_ok=True)

    b.plotDataExploration('figures/data_exploration/bird_data.png', 'figures/data_exploration/plane_data.png')
    b.plotVelLikelihoods('figures/likelihoods/combined_velocity_likelihoods.png')
    b.plotAccelLikelihoods('figures/likelihoods/bird_accel_likelihoods.png', 'figures/likelihoods/plane_accel_likelihoods.png', 'figures/likelihoods/combined_accel_likelihoods.png')
    b.plotTrainResults('figures/results/training_data_no_fe', 'figures/results/training_data_w_fe')
    b.plotTestResults('figures/results/test_data')


if __name__ == "__main__":
    main()