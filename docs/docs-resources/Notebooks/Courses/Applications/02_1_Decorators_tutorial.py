# %% [markdown]
"""
# Decorators Tutorial

François-David Collin (CNRS, IMAG, Paul-Valéry Montpellier 3
University)  
Wednesday, August 27, 2025

## Decorators

Decorators are a way to modify or extend the behavior of functions or
methods. They are a form of metaprogramming and can be used to add
functionality to existing functions or methods without modifying their
code. Decorators are a powerful tool in Python and are widely used in
libraries and frameworks.

In Python, decorators are implemented using the `@` symbol followed by
the decorator function name. Decorators can be used to add functionality
such as logging, timing, caching, access control, and more to functions
or methods.

In this tutorial, we will explore how decorators work and how to create
and use them in Python.

## Creating a Decorator

To create a decorator, we define a function that takes another function
as an argument and returns a new function that wraps the original
function. The new function can modify the behavior of the original
function by adding additional functionality before or after it is
called.

Here is an example of a simple decorator that prints a message before
and after calling a function:
"""


# %%
def my_decorator(func):
    def wrapper():
        print("Before calling the function")
        func()
        print("After calling the function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello, world!")

say_hello()

# %% [markdown]
"""
In this example, the `my_decorator` function takes a function `func` as
an argument and returns a new function `wrapper` that prints a message
before and after calling the original function. The `@my_decorator`
syntax is used to apply the decorator to the `say_hello` function.

## Decorator with Arguments

Decorators can also take arguments to customize their behavior. To
create a decorator with arguments, we need to define a function that
returns a decorator function. The decorator function then takes the
original function as an argument and returns a new function that wraps
the original function.

Here is an example of a decorator with arguments that prints a message
with a custom prefix before and after calling a function:
"""


# %%
def prefix_decorator(prefix):
    def decorator(func):
        def wrapper():
            print(f"{prefix}: Before calling the function")
            func()
            print(f"{prefix}: After calling the function")
        return wrapper
    return decorator

@prefix_decorator("INFO")
def say_hello():
    print("Hello, world!")

say_hello()

# %% [markdown]
"""
In this example, the `prefix_decorator` function takes an argument
`prefix` and returns a decorator function that prints a message with the
specified prefix. The `@prefix_decorator("INFO")` syntax is used to
apply the decorator with the prefix “INFO” to the `say_hello` function.

## Decorator Classes

Decorators can also be implemented using classes. To create a decorator
class, we define a class with a `__call__` method that takes the
original function as an argument and returns a new function that wraps
the original function.

Here is an example of a decorator implemented as a class that prints a
message before and after calling a function:
"""


# %%
class MyDecorator:
    def __init__(self, func):
        self.func = func

    def __call__(self):
        print("Before calling the function")
        self.func()
        print("After calling the function")

@MyDecorator
def say_hello():
    print("Hello, world!")

say_hello()

# %% [markdown]
"""
In this example, the `MyDecorator` class defines an `__init__` method
that takes the original function `func` as an argument and an `__call__`
method that wraps the original function. The `@MyDecorator` syntax is
used to apply the decorator to the `say_hello` function.

## Conclusion

Decorators are a powerful feature in Python that allow us to modify or
extend the behavior of functions or methods. They are widely used in
Python libraries and frameworks to add functionality such as logging,
timing, caching, access control, and more. By understanding how
decorators work and how to create and use them, we can write more
flexible and reusable code in Python.

In this tutorial, we explored how to create decorators, how to create
decorators with arguments, and how to create decorators using classes.
We also discussed some common use cases for decorators and how they can
be used to add functionality to existing functions or methods.

Decorators are a versatile tool in Python and can be used to solve a
wide range of problems. By mastering decorators, we can write more
concise, readable, and maintainable code in Python.

## References

-   [Python
    Decorators](https://docs.python.org/3/glossary.html#term-decorator)
-   [Real Python - Primer on Python
    Decorators](https://realpython.com/primer-on-python-decorators/)
-   [Python Wiki - Python
    Decorators](https://wiki.python.org/moin/PythonDecorators)
-   [Python Decorator
    Library](https://wiki.python.org/moin/PythonDecoratorLibrary)
-   [Python Decorator
    Tutorial](https://www.datacamp.com/community/tutorials/decorators-python)
-   [Python Decorator
    Examples](https://www.programiz.com/python-programming/decorator)
-   [Python Decorator
    Patterns](https://python-patterns.guide/gang-of-four/decorator-pattern/)
-   [Python Decorator Design
    Patterns](https://refactoring.guru/design-patterns/decorator/python/example)
-   [Python Decorator
    Cookbook](https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html)
-   [Python Decorator
    Recipes](https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html#recipes)
"""
