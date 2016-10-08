from cntk import cntk_py

class Function(cntk_py.Function):
    '''
    Base class of all operators.

    If it has only one output, one can invoke Variable methods on it, which it
    will relay to its only output.
    '''

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]

        if hasattr(self.output(), name):
            return getattr(self.output(), name)

        raise AttributeError("'%s' object has no attribute '%s'"%\
                (type(self), name))



