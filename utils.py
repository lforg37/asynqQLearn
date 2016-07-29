def emptyfunc():
    pass

class LockManager:
    def __init__(self, lockfunc, releasefunc, NeedsLock = True):
        self.__enter = emptyfunc if not NeedsLock else lockfunc
        self.__exit  = emptyfunc if not NeedsLock else releasefunc

    def __exit__(self, e_type, e_value, traceback):
        self.__exit()
    
    def __enter__(self):
        self.__enter()
