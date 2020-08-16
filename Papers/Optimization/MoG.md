# MoG-VQE: Multiobjective genetic variational quantum eigensolver Released: 10/07/2020, Read: 13/08/2020
----
Use a multiojective (don't know what this word means yet) genetic algortihm from the faily of evolutionary algorithms. 

## Genetic algorithms
The optimisation problem is described by a fitness function. Initialisation with a population of one or several individualsm which correspond to solutions. At each generation you perform variation (mutation) and selection based upon the fitness function. 

## MoG-VQE in particular
Uses genetic algorithm for btoh optimizing the VQE topology and optimizing the angles. Kind of like RotoSolve + RotoSelect.

### Topology optimization 
Two objective / fitness functions.
1. Energy
2. The number of 2 qubit gates
Utilise genetci algorithm NSGA-II. Searches for Pareto-optimal solutions, where one fitness function cannot be improved without degrading others. 

### Parameter optimization
CMA-ES (Covariance matrix adaptation evolution strategy) - they barely explain this so I suspect that they don't actually really use this. 

## Key takeaways
- Being able to efficiently reduce the number of CNOTs used is important














### References to read
[17] S. McArdle, T. Jones, S. Endo, Ying Li, S. Benjamin, and
Xiao Yuan, Variational ansatz-based quantum simulation
of imaginary time evolution, npj Quantum Information
5, 75 (2019).

[18] S. Endo, T. Jones, S. McArdle, Xiao Yuan, and S. Ben-
jamin, Variational quantum algorithms for discovering
Hamiltonian spectra, arXiv:1806.05707 (2018).

[19] J. Stokes, J. Izaac, N. Killoran, and G. Carleo, Quantum Natural Gradient, Quantum 4, 269 (2020).

[20] N. Yamamoto, On the natural gradient for variational quantum eigensolver, arXiv:1909.05074 (2019).

[21] B. Koczor and S. C. Benjamin, Quantum natural gradient generalised to non-unitary circuits, arXiv:191208660(2019).

[22] I. G. Ryabinkin, S. N. Genin, and A. F. Izmaylov,
Constrained Variational Quantum Eigensolver: Quantum Computer Search Engine in the Fock Space, J. Chem Theory. Comput. 15, 249 (2019).

[38] A. J. C. Woitzik, P. Kl. Barkoutsos, F. Wudarski, A.Buchleitner, and I. Tavernelli, Entanglement Production and Convergence Properties of the Variational Quantum Eigensolver, arXiv:2003.12490 (2020).