import time

from functools import wraps


def timer(f):
    """Decorator that times the amount of time it takes a function to
    complete and prints it to standard output"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        res = f(*args, **kwargs)
        tf = time.time()
        print(f"Time for function {f.__name__} to run: {tf - t0:.1f} seconds.")
        return res

    return wrapper
