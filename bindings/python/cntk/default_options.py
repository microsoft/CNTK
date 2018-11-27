# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

'''
Default options of CNTK functions. 


Usage: 

* ``with default_options():``, and
* ``with default_options_for():``
'''

# context manager for overriding defaults, use through default_options() or default_options_for() below
class _OptionsContextManager: # implement Python's 'with' protocol
    # this static member variable holds a linked list of default overrides, managed by _OptionsContextManager
    _current_default_overrides = None
    # constructor remembers the options-override record
    def __init__(self, scope, **kwargs):
        if '_scope' in kwargs or '_outer' in kwargs:
            raise ValueError("default_options: _scope or _outer are invalid (reserved) names.")
        self.scope = scope
        self.kwargs = kwargs
    # entering with block: link in a new default-options record at head
    def __enter__(self):
        from .variables import Record
        _OptionsContextManager._current_default_overrides = Record(_scope = self.scope, _outer = _OptionsContextManager._current_default_overrides, **self.kwargs) # insert new scope at head of link
        return self
    # exiting with block: restore previous remembered defaults
    def __exit__(self, type, value, traceback):
        _OptionsContextManager._current_default_overrides = _OptionsContextManager._current_default_overrides._outer # restore outer scope

# options scope without limit to specific functions, e.g.:
#   with default_options(activation=relu, init=he_normal(), pad=True):
#       model = Convolution((3,3), 32) >> Convolution((3,3), 64)  # will have relu activation and padding
def default_options(**kwargs):
    return _OptionsContextManager(None, **kwargs)

# options scope with limit to specific functions
# functions = function or list of functions
#   with default_options(activation=relu, init=he_normal()):
#        with default_options_for(Convolution, pad=True):
#            model = Convolution((3,3), 32) >> Convolution((3,3), 64) >> MaxPooling((2,2))  # Convolution will pad, MaxPooling won't
def default_options_for(functions, **kwargs):
    if not isinstance(functions, list):
        functions = [functions]
    return _OptionsContextManager(set(functions), **kwargs)

# simple wrapper to hold a default value
# meant to be both an indicator in a function signature that shows the default, e.g.:
#   def Convolution(args, init=default_override_or(glorot_uniform()), activation=default_override_or(identity), pad=default_override_or(False)):
class default_override_or:
    def __init__(self, value):
        self.value = value # This has a single member 'value' to hold the value.

# check if a parameter was given
# meant to be used inside functions that use this facility
# (No if it is still of type default_override_or.)
def is_default_override(value):
    return isinstance(value, default_override_or)

def get_default_override(function_or_class, **kwargs):
    '''
    Looks up an option default override.
    Meant to be used inside functions that use this facility.

    Args:
        function: the function that calls this.
          For example::

            def Convolution(args, init=default_override_or(glorot_uniform()), activation=default_override_or(identity), pad=default_override_or(False)):
            init = _get_default_override(Convolution, init=init) # pass default under the same name

    '''
    # parameter checking and casting
    if len(kwargs) != 1:
        raise TypeError("get_default_override() takes 1 keyword argument but %s were given" % len(kwargs))
    key, value = next(iter(kwargs.items())) # this is the keyword argument that the user passed in
    if function_or_class is not None:
        # first arg, unless None, must be an actual Python function...
        from inspect import isfunction, isclass
        if not (isfunction(function_or_class) or isclass(function_or_class)):
            raise ValueError('get_default_override() expects the first argument to be a Python function or class')
        # ...that has an arg with the same name as the given parameter
        from inspect import getargspec
        args, _, _, _ = getargspec(function_or_class) if isfunction(function_or_class) else getargspec(function_or_class.__init__)
        if key not in args:
            raise TypeError("{0}() has no argument named '{1}'".format(function_or_class.__name__, key))
    # if the value passed in is not a default, then use that value
    if not is_default_override(value):
        return value
    # traverse linked list of scopes inside-out until an override was found, else fall back to default
    opts = _OptionsContextManager._current_default_overrides
    while opts is not None:
        if opts._scope is None or function_or_class is None or function_or_class in opts._scope: # we are in the right scope
            if hasattr(opts, key):
                return opts[key]  # look up the option override and return it if present in this scope
        opts = opts._outer # step out one scope and try again
    return value.value # no override found: use the default as passed in

# In some case we couldn't use "with" expression, we need a global setting.
# This is the global dictionary with key / value format
_GlobalOptions = {}


# set a global option with the input initial value
def set_global_option(key, value):
    global  _GlobalOptions
    # overwrite previous setting
    _GlobalOptions[key] = value


# get the global option value, is not set yet, return the value user set in input
def get_global_option(key, default_value):
    if key in _GlobalOptions:
        return _GlobalOptions[key]
    else:
        # return the default value user passed in
        return default_value