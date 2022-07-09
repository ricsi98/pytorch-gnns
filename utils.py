

# ugly way to cache results on inpit parameter
def cache_result_in_param(fn):
    def wrapper(param):
        name = "__" + fn.__name__
        if not hasattr(param, name):
            res = fn(param)
            setattr(param, name, res)
        return getattr(param, name)
    return wrapper