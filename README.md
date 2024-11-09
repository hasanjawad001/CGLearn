## CGLearn: Consistent Gradient-based Learning for Out-of-Distribution Generalization

This source code is an implementation of the paper - [To be updated].

## Files and Folders

- **cglearn_linear.py**: Linear implementation of CGLearn.
- **cglearn_nonlinearMLP**: Nonlinear implementation of CGLearn (using Multi Layer Perceptron).
- **datasets/**: This folder contains two subfolders:
    - **datasets/linear/**: This subfolder contains a linearly generated dataset used for demonstration with the CGLearn linear implementation (cglearn_linear.py). For linear data generation, we followed a similar structural equation model as used in Invariant Risk Minimization. [Source code](https://github.com/facebookresearch/InvariantRiskMinimization). [Paper link](https://arxiv.org/abs/1907.02893v1).
    - **datasets/nonlinear/**: This subfolder contains the 'Yacht Hydrodynamics' dataset used for demonstration with the CGLearn Nonlinear implementation (cglearn_nonlinearMLP.py). The dataset is available from the [UCI repository](https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics).
