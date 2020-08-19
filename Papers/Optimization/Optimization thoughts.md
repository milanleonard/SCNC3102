# Optimization of variational circuits
----
1. Genetic approach for ansatz choice
2. Genetic approach for parameter values
3. Analytic expression using parameter shift method (2pi periodicity)
   1. Of the gradient (2 evaluations) (Gradient descent)
   2. Closed form solution of minimum (7 evaluations) (RotoSolve)

## Gradient descent
* All forms of classical optimization gradient descent available
* Quantum natural gradient - uses Fubini metric tensor to scale the gradients according to loss landscape