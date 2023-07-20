import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from scipy.stats import mannwhitneyu

def get_survival_function_value(i, sp, st):
    """
    Helper function to calculate survival function value
    """
    if len(st) < 1:
        return 1
    if i > np.max(st):
        return sp[-1]
    if i < np.min(st):
        return 1
    sind = np.argmax(st > i) - 1
    return sp[sind]

def load_data(filepath):
    """
    Load pickled data from a file
    """
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def reformat_data(data):
    """
    Reformat loaded data into a more manageable structure
    """
    reformatted_data = {}
    for i in range(50):
        reformatted_data[str(i)] = [data[f"{i}_sp1"], data[f"{i}_st1"], data[f"{i}_sp2"], data[f"{i}_st2"]]
    return reformatted_data

def calculate_survival_functions(D, T=10000):
    """
    Calculate survival functions based on loaded and reformatted data
    """
    split_num = len(D.keys())
    sf1m = np.zeros((split_num, T))
    sf2m = np.zeros((split_num, T))
    valid_split_count = 0

    for key in D.keys():
        [sp1, st1, sp2, st2] = D[key]
        if len(st1) > 2 and len(st2) > 2:
            for i in range(T):
                sf1m[valid_split_count, i] = get_survival_function_value(i, sp1, st1)
                sf2m[valid_split_count, i] = get_survival_function_value(i, sp2, st2)
            valid_split_count += 1

    print('The number of valid splits is ' + str(valid_split_count))
    sf1m = sf1m[:valid_split_count, :]
    sf2m = sf2m[:valid_split_count, :]
    asf1 = np.mean(sf1m, axis=0)
    asf2 = np.mean(sf2m, axis=0)

    return asf1, asf2, T

def plot_survival_functions(asf1, asf2, T):
    """
    Plot the survival functions
    """
    plt.figure()
    plt.plot(np.arange(0, T), asf1, color='blue', label='Predicted Low Risk')
    plt.plot(np.arange(0, T), asf2, color='orange',label='Predicted High Risk')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival')
    plt.legend()
    plt.show()

# Main program starts here
filepath = '/image_feature_your_path/'
data_filename = 'img'

# Load and reformat data
data = load_data(filepath + data_filename)
data = reformat_data(data)

# Calculate survival functions
asf1, asf2, T = calculate_survival_functions(data)

# Plot the survival functions
plot_survival_functions(asf1, asf2, T)
