# %% [markdown]
"""
# Multiprocessing in Python 3

Introduction to the `multiprocessing` module

François-David Collin (CNRS, IMAG, Paul-Valéry Montpellier 3
University)  
Wednesday, August 27, 2025

# Multiprocessing in Python 3

## Threads vs Processes

-   Thread
    -   Is bound to processor that python process running on
    -   Is controlled by Global Interpreter Lock (GIL)
        -   Single python bytecode executed at a time by any thread
-   Process
    -   Uses multiple processors
    -   Concurrency between threads and processes (local and remote)
    -   Ignores GIL
"""

# %%
from os import getpid, getppid
from time import sleep

def printer(val, wait=0):
    sleep(wait)
    print('Pid: {}, PPid: {}, Value: {}'
          .format(getpid(), getppid(), val))


# %% [markdown]
"""
## Process Instantiation

Let’s start with most basic example of spawning new process to run a
function
"""

# %%
from multiprocessing import Process

print('Starting demo...')
p = Process(target=printer, args=('hello demo',))
p.start()

# %% [markdown]
"""
### Process timing

-   Use printer’s delay to see process timing
-   Track multiple process objects
-   Execute code in main process while chile process is running
-   Use Process.join() to wait for processes to finish
"""

# %%
proc_list = []
for values in [('immediate', 0), ('delayed', 2), ('eternity', 5)]:
    p = Process(target=printer, args=values)
    proc_list.append(p)
    p.start()  # start execution of printer

print('Not waiting for proccesses to finish...')
    
[p.join() for p in proc_list]

print('After processes...')

# %% [markdown]
"""
## Process Pool

-   Worker processes instead of direct instantiation
-   Context manager to handle starting/joining child processes
-   Pool.map() works like default python `map(f, args)` function
-   Pool.map() Does not unpack args
"""

# %%
from multiprocessing.pool import Pool

with Pool(3) as pool:
    pool.map(printer, ['Its', ('A', 5), 'Race'])
    # each worker process executes one function

# %% [markdown]
"""
## Process + args/kwargs iteration with starmap
"""

# %%
with Pool(2) as pool:
    pool.starmap(printer, [('Its',), ('A', 2), ('Race',)])
    # one worker will execute 2 functions, one worker will execute the 'slow' function

# %% [markdown]
"""
## Thread Pool
"""

# %%
from multiprocessing.pool import ThreadPool

# Threadpool instead of process pool, same interface
with ThreadPool(2) as pool:
    pool.starmap(printer, [('Its', 5), ('A', 10), ('Race', 15)])

# %% [markdown]
"""
## Starmap is the bomb
"""


# %%
def pretend_delete_method(provider, vm_name):
    print('Pretend delete: {} on {}. (Pid: {})'
          .format(vm_name, provider, getpid()))    
    
# Assuming we fetched a list of vm names on providers we want to cleanup...
example_provider_vm_lists = dict(
    vmware=['test_vm_1', 'test_vm_2'],
    rhv=['test_vm_3', 'test_vm_4'],
    osp=['test_vm_5', 'test_vm_6'],
)

# %%
# don't hate me for nested comprehension here - building tuples of provider+name
from multiprocessing.pool import ThreadPool

# Threadpool instead of process pool, same interface
with ThreadPool(6) as pool:
    pool.starmap(
        pretend_delete_method, 
        [(key, vm) 
         for key, vms 
         in example_provider_vm_lists.items() 
         for vm in vms]
    )

# %% [markdown]
"""
## Locking

-   semaphore-type object that can be acquired and released
-   When acquired, only thread that has the lock can run
-   Necessary when using shared objects
"""


# %%
def not_safe_printing_method(provider, vm_name):
        print('Pretend delete: {} on {}. (Pid: {})'
              .format(vm_name, provider, getpid()))


# %%
with Pool(6) as pool:
    pool.starmap(
        not_safe_printing_method, 
        [(key, vm) for key, vms in example_provider_vm_lists.items() for vm in vms])

# %%
# Printing is thread safe, but will sometimes print separate messages on the same line (above)
# Use a lock around print
from multiprocessing import Lock, Manager

def safe_printing_method(lock, provider, vm_name):
    with lock:
        print('Pretend delete: {} on {}. (Pid: {})'
              .format(vm_name, provider, getpid()))


# %%
with Manager() as manager:
    lock = manager.Lock()
    with Pool(6) as pool:
        pool.starmap(
            safe_printing_method, 
            [(lock, key, vm) for key, vms in example_provider_vm_lists.items() for vm in vms])

# %% [markdown]
"""
# Queues

-   Store data/objects in child thread/processes and retrieve in parent
-   FIFO stack with put, get, and empty methods
"""

# %%
# Standard Queue
import queue
q = queue.Queue()
for x in range(4):
    q.put(x)
print("Members of the queue:")
y=z=q.qsize()

for n in list(q.queue):
    print(n, end=" ")
print("\nSize of the queue:")
print(q.qsize())

# %% [markdown]
"""
# Reminder on python serialization : “Pickling”

So what is pickling? Pickling is the serializing and de-serializing of
python objects to a byte stream. Unpicking is the opposite.

Pickling is used to store python objects. This means things like lists,
dictionaries, class objects, and more.
"""

# %%
import pickle # First, import pickle to use it

# %%
example_dict = {1:"6",2:"2",3:"f"} # we define an example dictionary, which is a Python object

# %%
pickle_out = open("dict.pickle","wb") # Next, we open a file (note that we open to write bytes in Python 3+)

# %%
pickle.dump(example_dict, pickle_out) # then we use pickle.dump() to put the dict into opened file, then close.

# %%
pickle_out.close() # and close(), it's very important to NOT forget to close your opened files.

# %% [markdown]
"""
The above code will save the pickle file for us, now we need to cover
how to access the pickled file:
"""

# %%
pickle_in = open("dict.pickle","rb") # Open the pickle file

# %%
example_dict = pickle.load(pickle_in) # Use pickle.load() to load it to a var.

# %% [markdown]
"""
That’s all there is to it, now you can do things like:
"""

# %%
print(example_dict)
print(example_dict[3])

# %% [markdown]
r"""
This shows that we’ve retained the dict data-type.

## Queues

-   multiprocessing.Queue
    -   **cannot be pickled** and thus can’t be passed to Pool methods
    -   can deadlock with improper join use
-   multiprocessing.Manager.Queue
    -   is proxy, can be pickled
    -   can be shared between processes

$\Longrightarrow$ prefer the use of managed queues

## Short example of queue use

In this example we share a managed queue between processes, and each
process can randomly put a boolean (indicating a failure for example) in
this queue.

This is our dummy function to parallelize, getting the shared queue as
an additional argument
"""

# %%
from random import randint

def multiple_output_method(provider, vm_name, fail_queue):
    # random success of called method
    if randint(0, 1):
        return True
    else:
        # Store our failure vm on the queue
        fail_queue.put(vm_name)
        return None


# %% [markdown]
"""
We need to instantiate the manager, and create a queue from it
"""

# %%
from multiprocessing import Manager

# Create instance of manager
manager = Manager()

# Create queue object to give to child processes
queue_for_failures = manager.Queue()

# %% [markdown]
"""
The (multi)-processing is there
"""

# %%
with Pool(2) as pool:
    results = pool.starmap(
        multiple_output_method, 
        [(key, vm, queue_for_failures)
         for key, vms
         in example_provider_vm_lists.items()
         for vm in vms]
    )

# %% [markdown]
"""
Now what’s up ?

Results :
"""

# %%
print('Results are in: {}'.format(results))

# %% [markdown]
"""
And what is in the queue ?
"""

# %%
failed_vms = []
# get items from the queue while its not empty
while not queue_for_failures.empty():
    failed_vms.append(queue_for_failures.get())
    
print('Failures are in: {}'.format(failed_vms))
