# %% [markdown]
"""
# Numba Introduction

François-David Collin (CNRS, IMAG, Paul-Valéry Montpellier 3
University)  
Wednesday, August 27, 2025

# Numba Basics

Numba is a just-in-time compiler of Python functions. It translates a
Python function when it is called into a machine code equivalent that
runs anywhere from 2x (simple NumPy operations) to 100x (complex Python
loops) faster. In this notebook, we show some basic examples of using
Numba.
"""

# %%
import numpy as np
import numba
from numba import jit

# %% [markdown]
"""
Let’s check which version of Numba we have:
"""

# %%
print(numba.__version__)

# %% [markdown]
"""
Numba uses Python *decorators* to transform Python functions into
functions that compile themselves. The most common Numba decorator is
`@jit`, which creates a normal function for execution on the CPU.

Numba works best on numerical functions that make use of NumPy arrays.
Here’s an example:
"""


# %%
@jit(nopython=True)
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    # assuming square input matrix
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting


# %% [markdown]
"""
The `nopython=True` option requires that the function be fully compiled
(so that the Python interpreter calls are completely removed), otherwise
an exception is raised. These exceptions usually indicate places in the
function that need to be modified in order to achieve better-than-Python
performance. We strongly recommend always using `nopython=True`.

The function has not yet been compiled. To do that, we need to call the
function:
"""

# %%
x = np.arange(100).reshape(10, 10)
go_fast(x)

# %% [markdown]
"""
This first time the function was called, a new version of the function
was compiled and executed. If we call it again, the previously generated
function executions without another compilation step.
"""

# %%
go_fast(2*x)

# %% [markdown]
"""
To benchmark Numba-compiled functions, it is important to time them
without including the compilation step, since the compilation of a given
function will only happen once for each set of input types, but the
function will be called many times.

In a notebook, the `%timeit` magic function is the best to use because
it runs the function many times in a loop to get a more accurate
estimate of the execution time of short functions.
"""

# %%
%timeit go_fast(x)

# %% [markdown]
"""
Let’s compare to the uncompiled function. Numba-compiled function have a
special `.py_func` attribute which is the original uncompiled Python
function. We should first verify we get the same results:
"""

# %%
print(np.testing.assert_array_equal(go_fast(x), go_fast.py_func(x)))

# %% [markdown]
"""
And test the speed of the Python version:
"""

# %%
%timeit go_fast.py_func(x)

# %% [markdown]
"""
The original Python function is more than 20x slower than the
Numba-compiled version. However, the Numba function used explicit loops,
which are very fast in Numba and not very fast in Python. Our example
function is so simple, we can create an alternate version of `go_fast`
using only NumPy array expressions:
"""


# %%
def go_numpy(a):
    return a + np.tanh(np.diagonal(a)).sum()


# %%
np.testing.assert_array_equal(go_numpy(x), go_fast(x))

# %%
%timeit go_numpy(x)

# %% [markdown]
"""
The NumPy version is more than 2x faster than Python, but still 10x
slower than Numba.

### Supported Python Features

Numba works best when used with NumPy arrays, but Numba also supports
other data types out of the box:

-   `int`, `float`
-   `tuple`, `namedtuple`
-   `list` (with some restrictions)
-   … and others. See the [Reference
    Manual](https://numba.pydata.org/numba-doc/latest/reference/pysupported.html)
    for more details.

In particular, tuples are useful for returning multiple values from
functions:
"""

# %%
import random

@jit(nopython=True)
def spherical_to_cartesian(r, theta, phi):
    '''Convert spherical coordinates (physics convention) to cartesian coordinates'''
    sin_theta = np.sin(theta)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * np.cos(theta)
    
    return x, y, z # return a tuple


# %%
@jit(nopython=True)
def random_directions(n, r):
    '''Return ``n`` 3-vectors in random directions with radius ``r``'''
    out = np.empty(shape=(n,3), dtype=np.float64)
    
    for i in range(n):
        # Pick directions randomly in solid angle
        phi = random.uniform(0, 2*np.pi)
        theta = np.arccos(random.uniform(-1, 1))
        # unpack a tuple
        x, y, z = spherical_to_cartesian(r, theta, phi)
        out[i] = x, y, z
    
    return out


# %%
random_directions(10, 1.0)

# %% [markdown]
"""
When Numba is translating Python to machine code, it uses the
[LLVM](https://llvm.org/) library to do most of the optimization and
final code generation. This automatically enables a wide range of
optimizations that you don’t even have to think about. If we were to
inspect the output of the compiler for the previous random directions
example, we would find that:

-   The function body for `spherical_to_cartesian()` was inlined
    directly into the body of the for loop in `random_directions`,
    eliminating the overhead of making a function call.
-   The separate calls to `sin()` and `cos()` were combined into a
    single, faster call to an internal `sincos()` function.

These kinds of cross-function optimizations are one of the reasons that
Numba can sometimes outperform compiled NumPy code.
"""
