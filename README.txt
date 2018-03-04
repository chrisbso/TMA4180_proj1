On the files of this project:
------------
**algos.py:
This file contains the unconstrained optimization algorithms, and their applications to model 1 and 2.
------------
**evaluate_f_gradf.py:
This file contains the algorithms for calculating the objective functions and its gradient for model 1 and 2.
------------
**generate_testproblem.py:
This file contains code for generating test problems.
In order to generate a test problem, set the parameters (n,m) of the main()-function of this file, then run it (n = # dimensions, m = # of points).
This will plot the relative error between the directional derivative and a finite difference approximation of it for a random direction, as well as plotting the solutions for n = 2 dimensions.
For more information, please see section 6 of the report.