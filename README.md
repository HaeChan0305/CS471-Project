# Augmentation of Graph Representations with Cycles

## Introduction
- Cyclic structures in molecules play a crucial role in characterizing molecules.
- Molecules are often modeled as a graph.
- However, message-passing GNN (MP-GNN) methods do not take cyclic information into account.
- Our approach allows MP-GNN models to use cycle information.

![Introduction](path/to/your/introduction-image.png)

## Related Works
- **Graph Convolutional Network (GCN)**: Used for message passing in graph-based ML tasks, but cycle information is lost.
- **Self-Attention Graph Pooling (SAGPool)**: A graph pooling method using self-attention, achieving superior graph classification performance on benchmarks.

![Related Works](path/to/your/related-works-image.png)

## Methods
Our approach consists of two main ideas:

1. **Cycle nodes (Green)**
   - Find cycles in the given graph and add them as nodes.
   - Represent the initial value to average of the nodes features in the cycle.

2. **Cycle size dimension (Pink)**
   - 0 for the original nodes & the length of cycle for the cycle nodes.

![Methods](path/to/your/methods-image.png)

## Theoretical Analysis

1. **Cycle nodes (Green)**
   - In the baseline method, the node representations of \(v\) and \(u\) given by MP-GNN are the same. In contrast, our method, which includes a cycle node (depicted as a red node), can differentiate between two non-isomorphic cycles.

2. **Cycle size dimension (Pink)**
   - In the aggregation step of GCN, the information about the number of neighbors of the cycle node, which corresponds to the length of the cycle, is lost because of the normalization term. Thus, we include the length of the cycle additionally.

![Theoretical Analysis](path/to/your/theoretical-analysis-image.png)

## Datasets

**TUDataset**: Graph data in a biomedical domain
1. **NC1**: molecule +/- effect on cell lung cancer
   - Node: Atom with one-hot vector of an atomic number
   - Edge: Bond between atoms
2. **NC109**: molecule +/- effect on ovarian cancer cell
3. **DHFR**: Dihydrofolate reductase enzyme
4. **BZR**: Ligands for Benzodiazepine receptor

![Datasets](path/to/your/datasets-image.png)

## Experiments

### 1. Experiments
   ![Experiments](path/to/your/experiments-image.png)

### 2. Results (iterations = 100)
   ![Results](path/to/your/results-image.png)

## Analysis

1. **(Table 1) Accuracy by models on NC1**
   - **Exp1 < Exp2 < Exp3 (p < .001)**
     - Cycle nodes allows the model to detect a cyclic structure
     - Cycle size dimension allows the model to distinguish cycle/node and cycle size

2. **Ablation (<0 Exp3 (p < .001)**
   - Adding 1-dimension without any information results in lower performance than ours.

3. **(Table 2) Accuracy by Baseline and Our best model on TUDatasets**
   - NC1, NC109, and DHFR: Baseline < Ours (p < .001)
   - BZR: Baseline < Ours (without statistical significant)

![Analysis](path/to/your/analysis-image.png)

## Conclusion

1. **Contributions**
   - We demonstrate GCN loses cyclic information theoretically and experimentally.
   - We propose a novel approach using cycle nodes and cycle size dimension.

2. **Further study**
   - A model that deals with cyclic structures in an end-to-end manner.
   - A better initialization scheme for cycle nodes.

![Conclusion](path/to/your/conclusion-image.png)

## References
1. Semi-Supervised Classification with Graph Convolutional Networks, Thomas N. Kipf and Max Welling, 2016
2. The Expressive Power of Graph Neural Networks, Haggai Maron and Heli Ben-Hamu, 2022
3. Self-Attention Graph Pooling, Lee, Jungwoo and Lee, Ryan A. Rossi and Kim, Sungchul and Ahmed, Nesreen, 2019
4. TUDataset: A collection of benchmark datasets for learning with graphs, Christopher Morris and Nils M. Kriege and Franka Bause and Kristian Kersting and Petra Mutzel and Marion Neumann, 2020

![References](path/to/your/references-image.png)
