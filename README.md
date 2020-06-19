# evolutionary_strategies
This repository provides an <b>unofficial</b> implementation of the evolutionary strategies optimizer by Beyer and Schwefel 2002<sup>1</sup>.
Evolutionary optimization is an optimization technique inspired by natural selection.
Most optimization techniques rely on analytical or numerical computations of function derivatives. Thus they are limited to smooth or at least continous functions. Gradient based methods usually also require convexity. With evolutionary algorithms even non-convex and discrete functions can be optimized.

<sup>1</sup> Beyer, H. G., & Schwefel, H. P. (2002). Evolution strategies–A comprehensive introduction. Natural computing, 1(1), 3-52.
## Requirements
- [x] `Python 3`
- [x] `Numpy`

## Usage
The example_optimization.ipynb jupiter-notebook demonstrates the basic functionality of the evolutionary strategies optimizer.
Over the course of 300 generations the population of 50 hypotheses evolves towards the correct solution within a reasonable precision.

### Optimization of a Simple Function Visualized
<p align="center">
<img src="https://github.com/janek-gross/evolutionary_strategies/blob/master/evolutionary_optimization.gif?raw=true" width="600" height="600"/>
</p>

## License
https://unlicense.org
