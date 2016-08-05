def emptyfunc():
    pass

class LockManager:
    def __init__(self, lockfunc, releasefunc, NeedsLock = True):
        self.__enter = lockfunc    if NeedsLock else emptyfunc
        self.__exit  = releasefunc if NeedsLock else emptyfunc

    def __exit__(self, e_type, e_value, traceback):
        self.__exit()
    
    def __enter__(self):
        self.__enter()
