import time
import functools

from Config import verbose

def clock(func):
    """ Decorator to time the execution of a function and print the result"""
    @functools.wraps(func)
    def clocked(self, *args):
        t0 = time.perf_counter()
        result = func(self, *args)
        elapsed = time.perf_counter() - t0
        if verbose:
            name = func.__name__
            arg_str = ', '.join(repr(arg) for arg in args)
            spaces = ' '*(40-len(name))
            print('          .. {} in{} --> {}s'.format(name, spaces,  round(elapsed, 4)))
        return result
    return clocked


def clock_noself(func):
    """ Decorator to time the execution of a function and print the result"""
    def clocked(*args):
        t0 = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        spaces = ' '*(40-len(name))
        print('            .. {} in{} --> {}s'.format(name, spaces,  round(elapsed, 4)))
        return result
    return clocked


def register(registry:list):
    """ Decorator to add a function to a given list of functions """
    def decorate(func):
        registry.append(func)
        return func
    return decorate




