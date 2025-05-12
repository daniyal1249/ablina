<div style="text-align: center;">
  <img src="ablina.svg" alt="ablina" style="width: 100%; height: auto;" />
</div>

## Documentation

https://ablina.readthedocs.io/en/latest


## Installation

Ablina can be installed using pip:

    pip install ablina

or by directly cloning the git repository:

    git clone https://github.com/daniyal1249/ablina.git

and running the following in the cloned repo:

    pip install .


## Overview

```python
>>> from ablina import *
```


### Define a Vector Space

To define a subspace of $\mathbb{R}^n$ or $\mathbb{C}^n$, use ``fn``

```python
>>> V = fn('V', R, 2)
>>> print(V)
```

    V (Subspace of R^2)
    -------------------
    Field      R
    Identity   [0, 0]
    Basis      [[1, 0], [0, 1]]
    Dimension  2
    Vector     [c0, c1]


You can provide a list of constraints 

```python
>>> U = fn('U', R, 2, constraints=['2*v0 == v1'])
>>> print(U)
```

    U (Subspace of R^2)
    -------------------
    Field      R
    Identity   [0, 0]
    Basis      [[1, 2]]
    Dimension  1
    Vector     [c0, 2*c0]


Or specify a basis 

```python
>>> W = fn('W', R, 2, basis=[[1, 2]])
>>> print(W)
```

    W (Subspace of R^2)
    -------------------
    Field      R
    Identity   [0, 0]
    Basis      [[1, 2]]
    Dimension  1
    Vector     [c0, 2*c0]


### Operations Involving Vectors

Check whether a vector is an element of a vector space 


```python
>>> [2, 4] in U
```

    True


```python
>>> [2, 3] in U
```

    False


Generate a vector from a vector space 


```python
>>> W.vector()
```

    [-2, -4]


```python
>>> W.vector(arbitrary=True)
```

    [c0, 2*c0]


Find the coordinate vector representation of a vector 


```python
>>> U.to_coordinate([2, 4])
```

    [2]


```python
>>> U.to_coordinate([2, 4], basis=[[4, 8]])
```

    [1/2]


Check whether a list of vectors is linearly independent 


```python
>>> V.are_independent([1, 2], [2, 3])
```

    True


```python
>>> V.are_independent([1, 2], [2, 4])
```

    False


### Operations on Vector Spaces

Check for equality of two vector spaces 


```python
>>> V == U
```

    False


```python
>>> U == W
```

    True


Check whether a vector space is a subspace of another 


```python
>>> U.is_subspace(V)
```

    True


```python
>>> V.is_subspace(U)
```

    False


Take the sum of two vector spaces 


```python
>>> X = V.sum(U)
>>> print(X)
```

    V + U (Subspace of R^2)
    -----------------------
    Field      R
    Identity   [0, 0]
    Basis      [[1, 0], [0, 1]]
    Dimension  2
    Vector     [c0, c1]


Take the intersection of two vector spaces 


```python
>>> X = V.intersection(U)
>>> print(X)
```

    V ∩ U (Subspace of R^2)
    -----------------------
    Field      R
    Identity   [0, 0]
    Basis      [[1, 2]]
    Dimension  1
    Vector     [c0, 2*c0]


Take the span of a list of vectors 


```python
>>> S = V.span('S', [1, -1])
>>> print(S)
```

    S (Subspace of R^2)
    -------------------
    Field      R
    Identity   [0, 0]
    Basis      [[1, -1]]
    Dimension  1
    Vector     [c0, -c0]


### Define a Linear Map