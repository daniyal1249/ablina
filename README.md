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
>>> V = fn('V', R, 3)
>>> print(V)
```

    V (Subspace of R^3)
    -------------------
    Field      R
    Identity   [0, 0, 0]
    Basis      [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Dimension  3
    Vector     [c0, c1, c2]


You can provide a list of constraints 

```python
>>> U = fn('U', R, 3, constraints=['v0 == 0', '2*v1 == v2'])
>>> print(U)
```

    U (Subspace of R^3)
    -------------------
    Field      R
    Identity   [0, 0, 0]
    Basis      [[0, 1, 2]]
    Dimension  1
    Vector     [0, c0, 2*c0]


Or specify a basis 

```python
>>> W = fn('W', R, 3, basis=[[1, 0, 0], [0, 1, 0]])
>>> print(W)
```

    W (Subspace of R^3)
    -------------------
    Field      R
    Identity   [0, 0, 0]
    Basis      [[1, 0, 0], [0, 1, 0]]
    Dimension  2
    Vector     [c0, c1, 0]


### Operations Involving Vectors

Check whether a vector is an element of a vector space 


```python
>>> [1, 2, 0] in W
```

    True


```python
>>> [1, 2, 1] in W
```

    False


Generate a random vector from a vector space 


```python
>>> U.vector()
```

    [0, 2, 4]


```python
>>> U.vector(arbitrary=True)
```

    [0, c0, 2*c0]


Find the coordinate vector representation of a vector 


```python
>>> W.to_coordinate([1, 2, 0])
```

    [1, 2]


```python
>>> W.from_coordinate([1, 2])
```

    [1, 2, 0]


```python
>>> W.to_coordinate([1, 2, 0], basis=[[1, 1, 0], [1, -1, 0]])
```

    [3/2, -1/2]


Check whether a list of vectors is linearly independent 


```python
>>> V.are_independent([1, 1, 0], [1, 0, 0])
```

    True


```python
>>> V.are_independent([1, 2, 3], [2, 4, 6])
```

    False


### Operations on Vector Spaces

Check for equality of two vector spaces 


```python
>>> U == W
```

    False


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
>>> X = U.sum(W)
>>> print(X)
```

    U + W (Subspace of R^3)
    -----------------------
    Field      R
    Identity   [0, 0, 0]
    Basis      [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Dimension  3
    Vector     [c0, c1, c2]


Take the intersection of two vector spaces 


```python
>>> X = U.intersection(W)
>>> print(X)
```

    U ∩ W (Subspace of R^3)
    -----------------------
    Field      R
    Identity   [0, 0, 0]
    Basis      []
    Dimension  0
    Vector     [0, 0, 0]


Take the span of a list of vectors 


```python
>>> S = V.span('S', [1, 2, 3], [4, 5, 6])
>>> print(S)
```

    S (Subspace of R^3)
    -------------------
    Field      R
    Identity   [0, 0, 0]
    Basis      [[1, 0, -1], [0, 1, 2]]
    Dimension  2
    Vector     [c0, c1, -c0 + 2*c1]


### Define a Linear Map