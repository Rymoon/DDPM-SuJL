from typing import Callable
import inspect
import types

def dargs_for_calling(f:Callable,d:dict):
    """Inspect the arg list of Callable f, fetch items for d and return;
    """
    import inspect
    args = inspect.getfullargspec(f)
    fal = [n for n in args.args]+[n for n in args.kwonlyargs]
    fd=  {}
    for k in fal:
        if k in d:
            fd[k] = d[k]
    
    return fd

def call_by_inspect(f:Callable,d:dict,**kwargs):
    """
    dargs = dargs_for_calling(f,d))

    dargs.update(kwargs)

    Return f(**d)
    """
    dargs = dargs_for_calling(f,d)
    dargs.update(kwargs)
    try:
        r = f(**dargs)   
    except Exception as e:
        f_ = f 
        fp = inspect.getsourcefile(f_)
        flineno = inspect.getsourcelines(f_)[1]
        print(f" - call_by_inspect, f defined at: {fp}, line {flineno}")
        raise e
    return r