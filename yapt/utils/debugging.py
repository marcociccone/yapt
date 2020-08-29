from time import time
from functools import wraps


class IllegalArgumentError(ValueError):
    pass


def timing(fn):
    @wraps(fn)
    def wrap(*args, **kwargs):
        ts = time()
        result = fn(*args, **kwargs)
        te = time()
        print('%r took: %2.4f sec' %  (fn.__name__, te-ts))
        return result
    return wrap


def call_counter(fn):
    @wraps(fn)
    def helper(*args, **kwargs):
        helper.calls += 1
        return fn(*args, **kwargs)
    helper.calls = 0
    helper.__name__ = fn.__name__

    return helper


def native(fn):
    @wraps(fn)
    def helper(*args, **kwargs):
        return fn(*args, **kwargs)
    helper.is_native = True
    helper.__name__ = fn.__name__
    return helper


def is_native(fn):
    return hasattr(fn, 'is_native') and fn.is_native
