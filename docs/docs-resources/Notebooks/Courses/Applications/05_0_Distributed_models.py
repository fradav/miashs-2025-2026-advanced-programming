# %% [markdown]
"""
# Distributed models examples

François-David Collin (CNRS, IMAG, Paul-Valéry Montpellier 3
University)  
Wednesday, August 27, 2025

# Prerequisites

You will need to install `dask[complete]` and `graphviz` packages.

# Initialization
"""

# %%
from dask.distributed import LocalCluster
cluster = LocalCluster(n_workers=8, threads_per_worker=1)          # Fully-featured local Dask cluster
client = cluster.get_client()
client

# %% [markdown]
"""
## Distributed prime numbers

Let’s revive our functions
"""

# %%
import math

def check_prime(n):
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


# %%
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# %%
import numpy as np

def find_primes(r):
    return np.array(list(filter(check_prime,r)))


# %% [markdown]
"""
### First steps

1.  Complete with the correct Dask code the function below which:
    -   takes a maximum number `N` and a `chunksize` as arguments
    -   creates a Dask `array` with numbers from 1 to `N` split into
        `chunksize` chunks
    -   maps the `find_primes` function to each chunk
    -   returns the resulting Dask array

See [Dask Array
documentation](https://docs.dask.org/en/stable/array.html).

``` python
import dask.array as da

def calculate_primes(N,chunksize):
    arr = ...
    return arr....
```

1.  Benchmark it for

``` python
N = 5000000
```
"""

# %%
N = 5000000

# %%
compute_graph = calculate_primes(N,8)
compute_graph

# %% [markdown]
r"""
1.  Visualize the computation graph

# An embarrassingly parallel example : distributed Monte-Carlo computing of $\pi$

If we sample randomly a bunch of $N$ points in the unity square, and
counts all points $N_I$ verifying the condition

$x^2 + y^2 \le 1$ which means they are in the upper right quarter of a
disk.

We have this convergence

$\lim_{N\to\infty} 4\frac{N_I}{N} = \pi$
"""

# %% [raw]
"""
<center>
"""

# %% [markdown]
"""
<img src="attachment:hpp2_0901.png" width="40%" />
"""

# %% [raw]
"""
</center>
"""

# %% [markdown]
r"""
### 2. Write the function which :

-   takes a number of estimates `nbr_estimates` as argument
-   samples them in the \[(0,0),(1,1)\] unity square
-   returns the number of points inside the disk quarter

``` python
def estimate_nbr_points_in_quarter_circle(nbr_estimates):
    ...
    return nbr_trials_in_quarter_unit_circle
```

### 3. Make it distributed

-   Wraps the previous function in \`\`\`python import dask.bag as db

    def calculate_pi_distributed(nnodes,nbr_samples_in_total) … return
    estimated_pi \`\`\`

-   `nnodes` will use only `nnodes` partitions for bag and split the
    number of estimates for each worker nodes into `nnodes` blocks.

-   Try it on `1e8` samples and benchmark it on 1 to 8 nodes. (use
    [`time`](https://docs.python.org/3/library/time.html#time.time))

-   Plot the performance gain over one node and comment the plot.

See [Dask Bag documentation](https://docs.dask.org/en/stable/bag.html).

$\Longrightarrow$ We see a near perfect linear scalability.
"""
