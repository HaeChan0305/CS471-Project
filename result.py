import scipy
import os
import re
import numpy as np
import matplotlib.pyplot as plt


# log_1.txt, log_2.txt, ...
def get_max_log_index(ablation):
    # if folder doesn't exist, make one
    if not os.path.exists(f'./logs/NCI1/ablation_{ablation}'):
        os.makedirs(f'./logs/NCI1/ablation_{ablation}')
    log_files = [f for f in os.listdir(f'./logs/NCI1/ablation_{ablation}') if re.match(r'log_\d+\.txt', f)]
    if not log_files:
        return 0
    max_index = max(int(re.search(r'\d+', f).group()) for f in log_files)
    return max_index


if __name__ == "__main__":
    # test accuracies are at the end of each file
    test_means = []
    test_stds = []
    for ablation in range(4):
        test_accuracies = []
        max_log_index = get_max_log_index(ablation)
        for i in range(1, max_log_index + 1):
            with open(f'./logs/NCI1/ablation_{ablation}' + '/' + f'log_{i}.txt', 'r') as f:
                lines = f.readlines()
                test_accuracy = float(lines[-1].split()[-1])
                test_accuracies.append(test_accuracy)

        test_accuracies = np.array(test_accuracies)

        # histogram of test accuracies
        plt.hist(test_accuracies, color='skyblue', edgecolor='black')
        plt.title('Histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

        # sharpio test
        statistic, pvalue = scipy.stats.shapiro(test_accuracies)
        print(f'Statistic: {statistic:.4f}')
        print(f'P-value: {pvalue:.4f}')


        # print average test accuracy and standard deviation
        mean = test_accuracies.mean()
        std = test_accuracies.std()
        
        print(f'Average test accuracy: {mean:.4f}')
        print(f'Standard deviation: {std:.4f}')
        test_means.append(mean)
        test_stds.append(std)