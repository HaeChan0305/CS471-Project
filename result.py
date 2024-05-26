import scipy
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse


# log_1.txt, log_2.txt, ...
def get_max_log_index(dataset, ablation):
    # if folder doesn't exist, make one
    if not os.path.exists(f'./logs/{dataset}/ablation_{ablation}'):
        os.makedirs(f'./logs/{dataset}/ablation_{ablation}')
    log_files = [f for f in os.listdir(f'./logs/{dataset}/ablation_{ablation}') if re.match(r'log_\d+\.txt', f)]
    if not log_files:
        return 0
    max_index = max(int(re.search(r'\d+', f).group()) for f in log_files)
    return max_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=int)
    args = parser.parse_args()
    experiment = args.experiment

    if experiment == 1:
        dataset = 'NCI1'
        test_results = []
        print("=============Results for each experiment=============")
        for ablation in [2, 1, 0, 3]:
            # if folder doesn't exist, skip
            if not os.path.exists(f'./logs/{dataset}/ablation_{ablation}'):
                continue
            test_accuracies = []
            max_log_index = get_max_log_index(dataset, ablation)
            for i in range(1, max_log_index + 1):
                with open(f'./logs/{dataset}/ablation_{ablation}' + '/' + f'log_{i}.txt', 'r') as f:
                    lines = f.readlines()
                    test_accuracy = float(lines[-1].split()[-1])
                    test_accuracies.append(test_accuracy)

            test_accuracies = np.array(test_accuracies)
            test_results.append(test_accuracies)

            if ablation <= 2:
                print(f'  Result for Exp {3-ablation}')
            else:
                print('  Result for ablation study')

            # print average test accuracy and standard deviation
            mean = test_accuracies.mean()
            std = test_accuracies.std()
            
            print(f'    Average test accuracy: {mean:.4f}')
            print(f'    Standard deviation: {std:.4f}')
        print("=============T-test results=============")
        for ablation_pair in [(2, 1), (1, 0), (3, 0)]:
            if ablation_pair == (2, 1):
                print('  Result for Exp 1 vs Exp 2')
            elif ablation_pair == (1, 0):
                print('  Result for Exp 2 vs Exp 3')
            elif ablation_pair == (0, 3):
                print('  Result for Ablation study vs Exp 3')
            statistic, pvalue = scipy.stats.ttest_ind(test_results[ablation_pair[0]], test_results[ablation_pair[1]], equal_var=False)
            print(f'    P-value: {pvalue}')
            if pvalue < 0.05:
                print('      Null hypothesis is rejected')
            else:
                print('      Null hypothesis is not rejected')

    elif experiment == 2:
        for dataset in ['NCI1', 'NCI109', 'DHFR', 'BZR']:
            print(f'============={dataset}=============')
            test_results = []
            # test accuracies are at the end of each file
            for ablation in [0, 2]:
                # if folder doesn't exist, skip
                if not os.path.exists(f'./logs/{dataset}/ablation_{ablation}'):
                    continue
                test_accuracies = []
                max_log_index = get_max_log_index(dataset, ablation)
                for i in range(1, max_log_index + 1):
                    with open(f'./logs/{dataset}/ablation_{ablation}' + '/' + f'log_{i}.txt', 'r') as f:
                        lines = f.readlines()
                        test_accuracy = float(lines[-1].split()[-1])
                        test_accuracies.append(test_accuracy)

                test_accuracies = np.array(test_accuracies)
                test_results.append(test_accuracies)

                if ablation == 0:
                    print('  Result for baseline')
                else:
                    print('  Result for our model')

                # print average test accuracy and standard deviation
                mean = test_accuracies.mean()
                std = test_accuracies.std()
                
                print(f'    Average test accuracy: {mean:.4f}')
                print(f'    Standard deviation: {std:.4f}')
            
            # t-test
            statistic, pvalue = scipy.stats.ttest_ind(test_results[0], test_results[1], equal_var=False)
            print('  Final t-test against the null hypothesis that our model is not better than the baseline')
            print(f'    P-value: {pvalue}')
            if pvalue < 0.05:
                print('      Null hypothesis is rejected')
            else:
                print('      Null hypothesis is not rejected')
    else:
        print('Invalid experiment number (should be 1 or 2)')