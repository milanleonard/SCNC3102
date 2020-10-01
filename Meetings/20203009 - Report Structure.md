# Report and where we are

# Gaps to fill in
* Go through and do the density matrix calculation, $\rho = U \rho U^\dag$
* Analytical description of the effects of errors

# What we've done
1. Benchmarked a bunch of optimizers on a toy model -- VQE on Helium atom (check atom) -- analytic and perfect
   1. The analytic optimizer RotoSolve dramatically outperformed all other optimizers -- single iteration to minimum -- only one local minimum -- monotonic decreasing cost function along each parameter 
2. Built a more complicated function -- QAOA -- looked at the cost landscape and made predictions about what we would expect.  
   1. Analytical derivation
   2. Sets the scene
3. Numerical simulation to demonstrate that the numerical technique is working -- do the calculations with no shot noise -- extend beyond three vertices. 
4. Shot noise
   1. Discussion about why some optimizers were effective against shot noise and why some completely failed to work
5. Depolarising noise
## Abstract
For the end

## Introduction
Motivation, essential elements of the background which distinguish this work from previous work, an aim statement, some methods, and what distinguishes this from other work.

## Background
### Brief intro to QC
Motivate hybrid. 
### Quantum noise

### Intro to variational quantum algorithms

### Optimization and initialisation -- parameterised learning

## Results and discussion
