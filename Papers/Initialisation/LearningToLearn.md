# Learning to learn with QNNs via NNs.
----
## Purpose of reading this paper
* Wanting to understand what a **class of variational algorithm** to benchmark performance looks like. 

## Interesting reference list **FILL ME OUT WITH NUMBERS**
- 28 (quantum form of backprop)
- 2,23 finite difference gradients. (30,31)
- 8, 32, 33 (QNN parameter initalization)
  - [8] L. Zhou, S.-T. Wang, S. Choi, H. Pichler, and M. D.Lukin, arXiv preprint arXiv:1812.01041 (2018).
  - [32] Z.-C. Yang, A. Rahmani, A. Shabani, H. Neven, and C. Chamon, Physical Review X 7, 021027 (2017)
  - [33] E. Grant, L. Wossnig, M. Ostaszewski,
and M. Benedetti, arXiv preprint arXiv:1903.05076 (2019)
- 10, class of ansatze known as quantum alternating operator ansatze.
- 29, optimization landscape of QNNs
  - [29] rXiv:1806.09729 (2018).J. R. Mcclean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, Nature Communications 9 (2018),
10.1038/s41467-018-07090-4.

## Abstract
* Meta-learning approach to rapidly find approximate optima in the parameter landscape for **classes of quantum variational algorithms**
* Goal is to **minimize number of queries to the cost function** <--> **minimize number of circuit evaluations**
* Focusses on the **initialisation** of the quantum parameters

## Introduction
* We train this RNN using random problem instances from specific classes of problems  -- QAOA for MaxCut, QAOA for Sherrington-Kirkpatrick Ising models, and VQE

## Quantum Classical Meta Learning
### Variational quantum algorithms
Already know all except that I'm not super comfortable Paulis that are at most k-local [49]

### Meta-learning with Neural  Optimizers
- Use an RNN to suggest parameter updates
- Interpet the QNN parameters and cost function evaluation over multiple quantum-classical iterations as a SEQ2SEQ learning problem
- One possible non-sparse loss function is the *cumulative regret*
- The loss function chosen is the *observed improvement* at teach time step
$$ L(\phi) = \mathbf{E}_{f,\mathbf{y}} \left[ \sum_{t=1}^T min\lbrace f(\theta_t) - min_{j<t}f(\theta_j),0\rbrace \right]$$
- Use a fixed number of iterations of the RNN as parameter initialisation for local search

## Results & Discusson
- Extremely effective as an initialisation strategy reducing the total number of evaluations