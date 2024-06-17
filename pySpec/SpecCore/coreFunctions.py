from functools import wraps
from copy import deepcopy


def inPlaceOp(func):
    """
    Decorator to give a class member function the option to either work in an in place or copy manner
    Add an optional inplace parameter to a class member function. If inplace is false (true is the default) it
    deepcopies the class and returns it.

    :param func: The function to be wrapped. Needs to be a class member function
    :return: The wrapped function
    """

    @wraps(func)
    # cls represents the self/instance, and always appears first. inplace is now inserted as an additional kwarg
    def wrapper(cls, *args, inplace: bool =True, **kwargs):
        if not inplace:
            cls = deepcopy(cls)

        return func(cls, *args, **kwargs)

    return wrapper
