# Initial Meeting 14/08/2020
### Milan, Marcus
----
#### Applications side of QB
- Not a lot has moved
  - Lots of tlak but not much actual progress

## Project
Phase one: Quantum optimization and initialisation

## Second project
----
- Two forms of noise from quantum component, statistical noise (shot noise), physical noise. 
- Input into the optimizer is noisy. Optimizer needs to be able accept noise.
- One strategy of doing that is by having a filter before the optimizer. Makes a statistical inference of the un-noisy output. Predicts the noise-free output. 
- It's a classical step, pre-processing step. Refined art that humans have mastered, stochastic filtering. 
- Quantum mechanical filtering might be possible

## Mathematical control theory - Barnett and Cameron

## Suggestion
1. Survey optimizers for ideal quantum computers. 
   1. Benchmark and visualize cost landscapes. 
2. How to handle noise. Introduce noise models into phase 1. 
   1. Modify the optimizer by adding some form of bayesian filtering. 

### Concerns
1. There's some thinking around on the filtering. 

## Receiving
* Paper from oakridge about noise models - characterizing noise models for different quantum computers. https://arxiv.org/abs/2001.08653