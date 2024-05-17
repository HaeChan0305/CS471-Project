import os
import argparse
import scipy
import numpy as np
import matplotlib.pyplot as plt


# parser = argparse.ArgumentParser()
# parser.add_argument('--ablation', type=int, help='0/1/2/3')
# parser.add_argument('--dataset', type=str, help='PROTEINS/NCI1')
# parser.add_argument('--validation', type=bool, default=True)
# args = parser.parse_args()

def extract_test_accuracy(file):
    with open(file, 'rb') as f:
        byte_data = f.read()
    str_data = byte_data.decode('utf-8')
    str_data = str_data.split("Test accuracy: ")[1].strip()
    test_accuracy = float(str_data)
    return test_accuracy

def check_validity(file):
    with open(file, "r") as f:
        try : 
            f.read().split("Test accuracy: ")[1].strip()
        except Exception as e:
            print(file, e)


if __name__ == "__main__":
    for dataset in ["NCI1"]:
        for ablation in [2,3]:
            dir = f'./logs_ablation{ablation}_{dataset}'
            # if args.validation:
            #     for file in os.listdir(dir):
            #         check_validity(os.path.join(dir, file)) 
    
            test_accuracys = [extract_test_accuracy(os.path.join(dir, file)) for file in os.listdir(dir)]
            # statistic, pvalue = scipy.stats.shapiro(test_accuracys)
            print(f"{dataset}-ablation{ablation}")
            print("mean: ", np.mean(test_accuracys))
            print("std: ", np.std(test_accuracys))
            print()
            print()
    
    
    # plt.hist(test_accuracys, color='skyblue', edgecolor='black')
    # plt.title('Histogram')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()

    
    
    
    
    