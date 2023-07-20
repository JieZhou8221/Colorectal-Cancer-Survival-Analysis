import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from utils import load_data, predict_image_features, calculate_survival, survival_function_value

def main():
    input_dir = 'input'
    output_dir = 'output'
    T = 10000

    testdata, clinical, model = load_data(input_dir)

    y_pre = predict_image_features(testdata, clinical, model)

    sf1, sf2 = calculate_survival(y_pre, clinical)

    D = {
        '0': [sf1.surv_prob, sf1.surv_times, sf2.surv_prob, sf2.surv_times]
    }

    splitnum = len(D.keys())
    sf1m = np.zeros((splitnum, T))
    sf2m = np.zeros((splitnum, T))

    for scnt in range(splitnum):
        [sp1, st1, sp2, st2] = D['0']
        if len(st1) > 2 and len(st2) > 2:
            for i in range(T):
                sf1m[scnt, i] = survival_function_value(i, sp1, st1)
                sf2m[scnt, i] = survival_function_value(i, sp2, st2)
    
    sf1m[0,0] = sf2m[0,0] = 1

    # Plotting average survival functions
    plt.figure()
    plt.plot(np.arange(0, T), np.mean(sf1m, axis=0), color='blue', label='Predicted Low Risk')
    plt.plot(np.arange(0, T), np.mean(sf2m, axis=0), color='orange', label='Predicted High Risk')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival')
    plt.legend()

    plt.savefig(join(output_dir, 'survival_plot.png'))
    plt.close()

if __name__ == "__main__":
    main()
