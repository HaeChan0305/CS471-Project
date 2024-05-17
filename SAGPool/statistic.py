import os
import argparse
import scipy
# import matplotlib
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--ablation', type=int, help='0/1/2/3')
parser.add_argument('--dataset', type=str, help='PROTEINS/NCI1')
parser.add_argument('--validation', type=bool, default=True)
args = parser.parse_args()

def extract_test_accuracy(file):
    with open(file, "r") as f:
        s = f.read().split("Test accuracy: ")[1].strip()
        test_accuracy = float(s)
    return test_accuracy

def check_validity(file):
    with open(file, "r") as f:
        try : 
            f.read().split("Test accuracy: ")[1].strip()
        except Exception as e:
            print(file, e)


if __name__ == "__main__":
    dir = f'./logs_ablation{args.ablation}_{args.dataset}'
    if args.validation:
        for file in os.listdir(dir):
            check_validity(os.path.join(dir, file)) 
    
    test_accuracys = [extract_test_accuracy(os.path.join(dir, file)) for file in os.listdir(dir)]
    
    plt.hist(test_accuracys, color='skyblue', edgecolor='black')
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    statistic, pvalue = scipy.stats.shapiro(test_accuracys)
    print("statistic: ", statistic)
    print("p-value: ", pvalue)
    
    
    
    