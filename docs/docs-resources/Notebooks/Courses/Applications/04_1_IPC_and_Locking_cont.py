# %% [markdown]
r"""
# Locking with `multiprocessing.Value`

François-David Collin (CNRS, IMAG, Paul-Valéry Montpellier 3
University)  
Wednesday, August 27, 2025

Take look at [Python Documentation on
`multiprocessing.Value`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Value)

Write a simple worker fonction which takes a `multiprocessing.Value` and
a `max_count` argument, and increment the value by one `max_count`
times:

``` python
import multiprocessing

def work(value, max_count):
    ...
```

1.  Write a main function instantiating an integer
    `multiprocessing.Value` initialized to zero and run `ncores` workers
    incrementing each `N` times the value one by one.
2.  Display the expected final value and the value calculated.
3.  Run it on `(8,100000)` (or even bigger). Replace `8` by the actual
    number of physical cores on the cpu you’re running on.

``` python
def run_workers(ncores,N):
    total_expected_count = ncores * N
    ...
```

> **Note**
>
> We may use a simplified version with `multiprocessing.Pool` and `map`
> to avoid the manual process management. However, to do so, we have to
> use the manager version of `multiprocessing.Value` which is (way)
> slower.

## Explanation

1.  Disassemble the worker function and try to look where locks occurs,
    according the reference documentation on `multiprocessing.Value`.
    The actual loading (resp. storing) of the value are done by
    `LOAD_ATTR` (resp. `STORE_ATTR`).  
    2. Explain the result

``` python
import dis

dis.dis(work)
```

``` plain
  5          12 LOAD_FAST                0 (value)
             14 DUP_TOP
                                                        #<--- Value lock acquired             
             16 LOAD_ATTR                1 (value)
                                                        #<--- Value lock released
             18 LOAD_CONST               1 (1)
             20 INPLACE_ADD
             22 ROT_TWO
                                                        #<--- Value lock acquired
             24 STORE_ATTR               1 (value)
                                                        #<--- Value lock released
             26 JUMP_ABSOLUTE            8
        >>   28 LOAD_CONST               0 (None)
             30 RETURN_VALUE
```

At instruction 18 (`18 LOAD_CONST`), nothing prevents another process to
load the (old) `value` attribute and be on instruction `18` too. Both
processes will proceed incrementing their private copy and writing it
back.

$\Rightarrow$ The result: the actual value got incremented only once,
not twice.

## Counter measure

Now, propose a solution. Use the reference documentation to modify the
`work` function, and the main function. Test it.

# Optimization

With the manual locking done now, is the native locking of
`multiprocessing.Value` still required ? Explain

<span class="proof-title">*Solution*. </span>As we already lock
the increment operation with both load and store of the value, the fine
grained locks of both operation is uneccessary.

We now want to use `multiprocessing.RawValue` which is devoid of any
lock mechanism, and a manual managed lock from
`multiprocessing.manager`.

Take a look at [Python Documentation on
`multiprocessing.RawValue`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.sharedctypes.RawValue)

1.  Write `work_rawlock` and `run_workers_rawlocked`, with careful
    consideration for where to instatiate the lock.
2.  Test it
3.  Benchmark and compare with the previous, print the speedup.
"""
