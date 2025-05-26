
```python
>>> from ablina import *
```


### Define a Vector Space

To define a subspace of R^n or C^n, use ``fn``

```python
>>> V = fn('V', R, 3)
>>> print(V.info())
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
>>> print(U.info())
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
>>> W = fn('W', R, 3, basis=[M[1, 0, 0], M[0, 1, 0]])
>>> print(W.info())
```

    W (Subspace of R^3)
    -------------------
    Field      R
    Identity   [0, 0, 0]
    Basis      [[1, 0, 0], [0, 1, 0]]
    Dimension  2
    Vector     [c0, c1, 0]


### Operations with Vectors

Check whether a vector is an element of a vector space 


```python
>>> M[1, 2, 0] in W
```

    True


```python
>>> M[1, 2, 1] in W
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
>>> W.to_coordinate(M[1, 2, 0])
```

    [1, 2]


```python
>>> W.from_coordinate(M[1, 2])
```

    [1, 2, 0]


```python
>>> W.to_coordinate(M[1, 2, 0], basis=[M[1, 1, 0], M[1, -1, 0]])
```

    [3/2, -1/2]


Check whether a list of vectors is linearly independent 


```python
>>> V.is_independent(M[1, 1, 0], M[1, 0, 0])
```

    True


```python
>>> V.is_independent(M[1, 2, 3], M[2, 4, 6])
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
>>> V.is_subspace(U)
```

    True


```python
>>> U.is_subspace(V)
```

    False


Take the sum of two vector spaces 


```python
>>> X = U.sum(W)
>>> print(X.info())
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
>>> print(X.info())
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
>>> S = V.span('S', M[1, 2, 3], M[4, 5, 6])
>>> print(S.info())
```

    S (Subspace of R^3)
    -------------------
    Field      R
    Identity   [0, 0, 0]
    Basis      [[1, 0, -1], [0, 1, 2]]
    Dimension  2
    Vector     [c0, c1, -c0 + 2*c1]


### Define a Linear Map

```python
>>> def mapping(vec):
>>>     return M[vec[0], vec[1], 0]
>>>
>>> T = LinearMap('T', domain=V, codomain=W, mapping=mapping)
>>> print(T.info())
```

    T : V → W
    ---------
    Field        R
    Rank         2
    Nullity      1
    Injective?   False
    Surjective?  True
    Bijective?   False
    Matrix       [[1, 0, 0], [0, 1, 0]]


```python
>>> T(M[0, 0, 0])
```

    [0, 0, 0]


```python
>>> T(M[1, 2, 3])
```

    [1, 2, 0]


### Operations with Linear Maps

Find the image of a linear map 


```python
>>> im = T.image()
>>> print(im.info())
```

    im(T) (Subspace of R^3)
    -----------------------
    Field      R
    Identity   [0, 0, 0]
    Basis      [[1, 0, 0], [0, 1, 0]]
    Dimension  2
    Vector     [c0, c1, 0]


Find the kernel of a linear map 


```python
>>> ker = T.kernel()
>>> print(ker.info())
```

    ker(T) (Subspace of R^3)
    ------------------------
    Field      R
    Identity   [0, 0, 0]
    Basis      [[0, 0, 1]]
    Dimension  1
    Vector     [0, 0, c0]


### Define an Inner Product

Here we define the standard dot product 

```python
>>> def mapping(vec1, vec2):
>>>     return sum(i * j for i, j in zip(vec1, vec2))
>>>
>>> dot = InnerProduct('dot', vectorspace=V, mapping=mapping)
>>> print(dot.info())
```

    dot : V × V → R
    ---------------
    Orthonormal Basis  [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Matrix             [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


```python
>>> dot(M[1, 2, 3], M[1, 2, 3])
```

    14


### Operations with Inner Products

Compute the norm of a vector 


```python
>>> dot.norm(M[1, 2, 3])
```

    sqrt(14)


Check whether a list of vectors is pairwise orthogonal


```python
>>> dot.is_orthogonal(M[1, 2, 3], M[4, 5, 6])
```

    False


```python
>>> dot.is_orthogonal(M[0, 0, 0], M[1, 2, 3])
```

    True


Check whether a list of vectors is orthonormal 


```python
>>> dot.is_orthonormal(M[1, 0, 0], M[0, 1, 0], M[0, 0, 1])
```

    True


Take the orthogonal complement of a vector space 

```python
>>> X = dot.ortho_complement(U)
>>> print(X.info())
```

    perp(U) (Subspace of R^3)
    -------------------------
    Field      R
    Identity   [0, 0, 0]
    Basis      [[1, 0, 0], [0, 1, -1/2]]
    Dimension  2
    Vector     [c0, c1, -c1/2]


### Define a Linear Operator

```python
>>> def mapping(vec):
>>>     return M[vec[0], 2*vec[1], 3*vec[2]]
>>>
>>> T = LinearOperator('T', vectorspace=V, mapping=mapping)
>>> print(T.info())
```

    T : V → V
    ---------
    Field        R
    Rank         3
    Nullity      0
    Injective?   True
    Surjective?  True
    Bijective?   True
    Matrix       [[1, 0, 0], [0, 2, 0], [0, 0, 3]]


```python
>>> T(M[1, 1, 1])
```

    [1, 2, 3]


### Operations with Linear Operators

Given an inner product, check whether a linear operator 


```python
>>> T.is_symmetric(dot)
```

    True


```python
>>> T.is_hermitian(dot)
```

    True


```python
>>> T.is_orthogonal(dot)
```

    False


```python
>>> T.is_unitary(dot)
```

    False


```python
>>> T.is_normal(dot)
```

    True
