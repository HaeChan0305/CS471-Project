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
    with open(file, 'r') as f:
        s  = f.read()
        s = s.split("Test accuracy: ")[1].strip()
        test_accuracy = float(s)
    return test_accuracy

def check_validity(file):
    with open(file, "r") as f:
        try : 
            f.read().split("Test accuracy: ")[1].strip()
        except Exception as e:
            print(file, e)


if __name__ == "__main__":
    for dataset in ["PROTEINS", "NCI1"]:
        for ablation in range(4):
            dir = f'./logs/{dataset}/ablation_{ablation}'
            # if args.validation:
            #     for file in os.listdir(dir):
            #         check_validity(os.path.join(dir, file)) 
    
            test_accuracys = [extract_test_accuracy(os.path.join(dir, file)) for file in os.listdir(dir)]
            
            # NCI1-ablation3
            # test_accuracys = [0.7347931873479319, 0.7737226277372263, 0.6739659367396593, 0.7493917274939172, 0.708029197080292, 0.7201946472019465, 0.7712895377128953, 0.6642335766423357, 0.635036496350365, 0.7591240875912408, 0.6934306569343066, 0.7274939172749392, 0.7761557177615572, 0.7104622871046229, 0.6861313868613139, 0.7396593673965937, 0.7518248175182481, 0.7007299270072993, 0.754257907542579, 0.7104622871046229, 0.7274939172749392, 0.7347931873479319, 0.7493917274939172, 0.6690997566909975, 0.7445255474452555, 0.754257907542579, 0.7493917274939172, 0.7031630170316302, 0.732360097323601, 0.7226277372262774]

            # NCI1-ablation2
            # test_accuracys = [0.7250608272506083, 0.6763990267639902, 0.6909975669099757, 0.6909975669099757, 0.6958637469586375, 0.6739659367396593, 0.7274939172749392, 0.7299270072992701, 0.7128953771289538, 0.7688564476885644, 0.7591240875912408, 0.683698296836983, 0.7372262773722628, 0.6545012165450121, 0.6861313868613139, 0.708029197080292, 0.7226277372262774, 0.6739659367396593, 0.6763990267639902, 0.7128953771289538, 0.6447688564476886, 0.6934306569343066, 0.6788321167883211, 0.7031630170316302, 0.7055961070559611, 0.5766423357664233, 0.7396593673965937, 0.6861313868613139, 0.754257907542579, 0.7177615571776156]
            
            print(f"{dataset}-ablation{ablation}")
            print("mean: ", np.mean(test_accuracys))
            print("std: ", np.std(test_accuracys, ddof=1))
            print()
            print()
    
    
    # plt.hist(test_accuracys, color='skyblue', edgecolor='black')
    # plt.title('Histogram')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()

    
    
    
    
    