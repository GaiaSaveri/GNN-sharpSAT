## Graph Neural Networks for Propositional Model Counting

Gaia Saveri, Luca Bortolussi (2022)

#### Abstract

Graph Neural Networks (GNNs) have been recently leveraged to solve several logical reasoning tasks. Nevertheless, counting problems such as propositional model counting (#SAT) are still mostly approached with traditional solvers. Here we tackle this gap by presenting an architecture based on the GNN framework for belief propagation (BP) of Kuch et al., extended with self-attentive GNN and trained to approximately solve the #SAT problem. We ran a thorough experimental investigation, showing that our model, trained on a small set of random Boolean formulae, is able to scale effectively to much larger problem sizes, with comparable or better performances of state of the art approximate solvers. Moreover, we show that it can be efficiently fine-tuned to provide good generalization results on different formulae distributions, such as those coming from SAT-encoded combinatorial problems.

### Dependencies

---

In order to run the code, the following libraries should be installed in this folder:

* `PyMiniSolvers`, which is the Python interface for the exact SAT solver Minisat; it can be installed from http://minisat.se/MiniSat.html

* `sharpSAT`, which is the exact #SAT solver sharpSAT; it can be installed from https://github.com/marcthurley/sharpSAT

* `approxmc`, which is the approximate #SAT solver ApproxMC; it can be installed from https://github.com/meelgroup/approxmc

### Executing the Scripts

---

The folder `scripts` contains the bash scripts to generate the datasets and reproduce the results of the paper. These should be either executed inside that folder, or all the paths to files and directories should be adapted accordingly.


### Acknowledgements

---

Some parts of the code, in particular those regarding data preprocessing and batching, are re-adapted for this framework from https://github.com/jkuck/BPNN.
