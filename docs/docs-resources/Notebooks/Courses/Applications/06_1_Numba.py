# %% [markdown]
"""
# Numba first steps

François-David Collin (CNRS, IMAG, Paul-Valéry Montpellier 3
University)  
Wednesday, August 27, 2025

## Rewind on sequence searching

Make a proper function of the sequence searching Application from
`0_Numpy workout.ipynb`. Test it.

``` python
def search_sequence_numpy(data,seq):
    ...
```
"""

# %%
import numpy as np

data = np.array([1,3,2,3,5,9,2,3,5,1,0],dtype=np.uint8)
sequence = np.array([3,5],dtype=np.uint8)

# %%
search_sequence_numpy(data,sequence)

# %% [markdown]
"""
## Numba migration

We want to “unroll” numpy code in double nested loop, simply walking
`data` and `sequence` and put results in an accumulator, one element at
a time. Write the missing line.

``` python
import numba

@numba.jit(nopython=True)
def search_sequence_numba(data,seq):
    cor_size = data.size-seq.size+1
    
    matches = np.ones(cor_size,dtype=np.uint8)
    
    for i in range(cor_size): # walking on data
        for j in range(seq.size): # walking on sequence
            ...
            
    return np.nonzero(matches)[0]

search_sequence_numba(data,sequence)
```

## Blow it up

Generate 10000 of random digits (the data) a sequence of 3 digits, and
benchmark both versions (numpy and numba) on it. Compare and comment.
"""

# %%
search_sequence_numpy(data_rand,sequence_rand)

# %%
import pandas as pd

benchmarks = pd.DataFrame(
    {"data size":int(1e6),
     "version":"numpy",
     "timing":numpy_time.average},
    index=[0])
benchmarks = pd.concat(
    [benchmarks,
     pd.Series(
         {"data size":int(1e6),
          "version":"numba",
          "timing":numba_time.average
         }).to_frame().T],ignore_index=True)

# %% [markdown]
"""
## And now… parallelize

Question : is pattern matching like we just did an “embarrassingly
parallel” problem ? Explain.

It shouldn’t be : if we partition the data in chunks, the pattern
matching will miss any match occuring between two consecutive chunks.

Numba got a powerful (multi-threaded) parallelization feature, one
just needs to : 1. add `parallel=True` in the decorator call 2. replace
python `range` used for looping with numba’s `prange`.

With a spetial attention to where you could put parallelization
directive with prange (remember the “Concepts” course). Test and
benchmark, give the speedup and comment.

Why there si no *race condition* there ? (Tip : consider concurrent
access in multi-threading, and look closely in the loop to read/store to
the data).

Over 2~3 speedup over the non-parallel version is a sensible one on a
the current 4-core CPU.

There is no race condition because all data/sequence access are only
read and the only assignment is on `matches[i]` which depends only
on itself and data/sequence read. As a `prange` on `matches` index gives
exclusive partitions per thread, it is guaranted that a thread will
never access `matches` from other thread partitions.

# Multi-processing vs Multi-threading

Is this type of parallelization “trick” also possible as is with
multi-processing ?

Has multi-threading any advantage over multiprocessing in this context ?

Let’s look into it.

Make a modified `search_sequence_numba2` which takes a index subrange of
the `matches` array and return the matches only on this range. Test it
on the original `data` and `sequence` with two chunks.

``` python
@numba.jit(nopython=True)
def search_sequence_numba2(data,seq,chunk):
    matches = ...
    
    for i,ic in enumerate(chunk): # walking on data
        for j in range(seq.size): # walking on sequence
            ...
            
    return np.nonzero(matches)[0]+chunk[0]
```

Recall the chunks function generator
"""


# %%
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# %% [markdown]
"""
Now make the multiprocessing version, test it. Benchmark it and give the
speedup over the numba parallel version.

Do it again but this time for a random data with 10 millions of digits.

``` python
import multiprocessing
from itertools import chain

def search_sequence_multiprocessing(data,seq,ncores):
    cor_size = data.size-seq.size+1
    
    ...
```
"""

# %%
benchmarks = pd.concat(
    [benchmarks,
     pd.DataFrame([
         {"data size":int(1e7),"version":"numpy","timing":numpy10M_time.average},
         {"data size":int(1e7),"version":"numba","timing":numba10M_time.average},
         {"data size":int(1e7),"version":"numba parallel","timing":numba_parallel10M_time.average},
         {"data size":int(1e7),"version":"multiprocessing numba","timing":multiprocessing_numba10M_time.average}
     ])],
     ignore_index=True)

# %% [markdown]
"""
# Market it with a chart

Make a bar chart with all versions timings, taking the numpy version as
reference, and both (1e6, 1e7) runs of the data.
"""
