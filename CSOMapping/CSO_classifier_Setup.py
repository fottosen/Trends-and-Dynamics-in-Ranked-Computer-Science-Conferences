from cso_classifier import CSOClassifier
from pprint import pprint

"""
### Setup

CSOClassifier.setup()
exit() # it is important to close the current console, to make those changes effective

### Update
CSOClassifier.update()

### Test

import cso_classifier as test
test.test_classifier_single_paper() # to test it with one paper
test.test_classifier_batch_mode() # to test it with multiple papers

"""
t="Parallel and distributed random coordinate descent method for convex error bound minimization."
a="In this paper we propose a parallel and distributed random (block) coordinate descent method for minimizing the sum of a partially separable smooth convex function and a fully separable non-smooth convex function. In this algorithm the iterate updates are done independently and thus it is suitable for parallel and distributed computing architectures. We prove linear convergence rate for the proposed algorithm on the class of problems satisfying a generalized error bound property. We also show that the theoretical estimates on the convergence rate depend on the number of blocks chosen randomly and a natural measure of separability of the objective function. Numerical simulations are also provided to confirm our theory."

# st=' '.join(sorted(t.split(" "), key=str.lower))
# sa=' '.join(sorted(a.split(" "), key=str.lower))

paper = {
        "doi": {
                "title": t,
                "abstract": a,
        }
}


cc = CSOClassifier(workers = 1, modules = "both", enhancement = "all", explanation = False, delete_outliers=True)
result = cc.batch_run(paper)
pprint(result)

"""

>>> pprint(result)
### NON SORTED
{'enhanced': ['distributed computer systems',
              'evolutionary algorithms',
              'particle swarm optimization (pso)',
              'optimization problems',
              'approximation theory',
              'echo suppression',
              'software'],
 'explanation': {'adaptive algorithms': ['algorithm'],
                 'approximation theory': ['convergence', 'convergence rate'],
                 'coordinate descent': ['coordinate descent'],
                 'distributed computer systems': ['distributed computing'],
                 'distributed computing': ['distributed computing'],
                 'echo suppression': ['coordinate descent'],
                 'evolutionary algorithms': ['algorithm'],
                 'hybrid algorithms': ['algorithm'],
                 'nonconvex': ['convex'],
                 'optimization problems': ['convex'],
                 'particle swarm optimization (pso)': ['algorithm'],
                 'rate of convergence': ['convergence', 'convergence rate'],
                 'software': ['algorithm']},
 'semantic': ['nonconvex',
              'hybrid algorithms',
              'adaptive algorithms',
              'rate of convergence',
              'coordinate descent'],
 'syntactic': ['distributed computing', 'coordinate descent'],
 'union': ['distributed computing',
           'hybrid algorithms',
           'nonconvex',
           'rate of convergence',
           'coordinate descent',
           'adaptive algorithms']}

           
### SORTED
{'enhanced': ['numerical methods',
              'evolutionary algorithms',
              'particle swarm optimization (pso)',
              'optimization problems',
              'approximation theory',
              'software'],
 'explanation': {'adaptive algorithms': ['algorithm', 'algorithm algorithm'],
                 'approximation theory': ['convergence convergence',
                                          'convergence'],
                 'error estimates': ['error estimates'],
                 'evolutionary algorithms': ['algorithm',
                                             'algorithm algorithm'],
                 'hybrid algorithms': ['algorithm', 'algorithm algorithm'],
                 'nonconvex': ['convex convex', 'convex'],
                 'numerical methods': ['error estimates'],
                 'optimization problems': ['convex convex', 'convex'],
                 'particle swarm optimization (pso)': ['algorithm',
                                                       'algorithm algorithm'],
                 'rate of convergence': ['convergence convergence',
                                         'convergence'],
                 'software': ['algorithm', 'algorithm algorithm']},
 'semantic': ['error estimates',
              'nonconvex',
              'hybrid algorithms',
              'adaptive algorithms',
              'rate of convergence'],
 'syntactic': ['error estimates'],
 'union': ['error estimates',
           'hybrid algorithms',
           'nonconvex',
           'rate of convergence',
           'adaptive algorithms']}

"""