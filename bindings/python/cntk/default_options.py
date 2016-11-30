# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# default_options: with default_options() pattern

from cntk.utils import Record

# This record contains the defaults for a number of optional parameters to layers.
# These can be overwritten temporarily by saying
#    with default_options(init=..., ...):
#        # code block within which the changed defaults are active
# TODO: rename to _current_default_overrides
_current_default_options = None
#Record(
    #init=glorot_uniform(),
    ##activation=None,                  # Dense() and Convolution() have no activation by default
    #pad=False, # BUGBUG: not done for pooling at present. Need a special default? How to name?
    ## ^^ This should be addressed by allowing configs per layer type.
    ##    To be addressed as a per-layer default. See default_options below.
    #bias=True,
    #init_bias=0,
    #enable_self_stabilization=False,  # Stabilizer() and LSTM()
    #initial_state=None,               # Recurrence()
    #use_peepholes=False,              # LSTM()
    #dtype=np.float32,                 # Constant(), Parameter(), Input()
#    _scope = None,                    # set of all functions that this scope belongs to; None means "all"
#    _outer = None                     # link to outer scope
#)

# simple wrapper to hold default values#
# meant to be both an indicator in a function signature and a detectable wrapper of a default
# This has a single member 'value' to hold the value.
class default_override_or:
    def __init__(self, value):
        self.value = value

# check if a parameter was given
# (No if it is still of type default_override_or.)
def is_default_override(value):
    return isinstance(value, default_override_or)

# look up an option default override
# 'function' is the function that calls this
# new pattern:
# def Convolution(args, init=default_override_or(glorot_uniform()), activation=default_override_or(identity), pad=default_override_or(False)):
#     init = _get_default_override(Convolution, init=init) # pass default under the same name
def get_default_override(function, **kwargs):
    # parameter checking and casting
    if len(kwargs) != 1:
        raise ValueError("_get_default_override() expects 1 keyword argument")
    key, value = [kvp for kvp in kwargs.items()][0] # this is the keyword argument that the user passed in  --TODO: can this be simplified?
    from inspect import signature, isfunction  # check if key is a valid parameter  --TODO: should check for kw arguments
    if not isfunction(function):
        raise ValueError('First argument must be a function')
    try:
        signature(function).parameters[key]
    except:
        raise TypeError("{0}() has no argument named '{1}'".format(function.__name__, key))
    # if the value passed in is not a default, then use that value
    if not is_default_override(value):
        return value
    # traverse linked list of scopes inside-out until an override was found, else fall back to default
    opts = _current_default_options
    while opts is not None:
        if opts._scope is None or function in opts._scope: # we are in the right scope
            if hasattr(opts, key):
            #try:
                value = opts[key]  # look up the option override and return it if present in this scope
                return value
            #except:
            #    pass       # no such override in this scope
        opts = opts._outer # step out one scope and try again
    return value.value # no override found: use the default as passed in

#_default_sentinel           = '(default)'           # This is a singleton sentinel value we recognize and replace in _initializer_for()
#_default_sentinel_init      = '(init default)'      # use different ones for init andinit_bias so we can distinguish them in _initializer_for()
#_default_sentinel_init_bias = '(init_bias default)'
# in function signatures we use symbols that indicate the default default in their name
#init_default_or_glorot_uniform             = _default_sentinel_init
#activation_default_or_None                 = _default_sentinel
#init_bias_default_or_0                     = _default_sentinel_init_bias
#bias_default_or_True                       = _default_sentinel
#pad_default_or_False                       = _default_sentinel
#enable_self_stabilization_default_or_False = _default_sentinel
#initial_state_default_or_None              = _default_sentinel
#use_peepholes_default_or_False             = _default_sentinel
#dtype_default_or_float32                   = _default_sentinel_init

# check whether a parameter is a default
# This is meant to be used by implementations of layers that take default values that may default to default-defaults.
#def _is_given(p):
#    return p is not _default_sentinel and p is not _default_sentinel_init and p is not _default_sentinel_init_bias

# scope guard for overriding defaults
# with default_options(activation=relu, init=he_normal(), pad=True):
#     model = Convolution((3,3), 32) >> Convolution((3,3), 64)  # will have relu activation and padding
# to limit it to a specific operation or set of operations, use e.g.
# with default_options(activation=relu, init=he_normal()):
#     with default_options_for(Convolution, pad=True):
#         ...
class _OptionsStack: # implement Python's 'with' protocol
    # constructor remembers the options-override record
    def __init__(self, scope, **kwargs):
        self.scope = scope
        self.kwargs = kwargs
    # entering with block: link in a new default-options record at head
    def __enter__(self):
        global _current_default_options
        #self.new_scope._outer = _current_default_options # remember outer link into this objects scope (for traversal and __exit__())
        _current_default_options = Record(_scope = self.scope, _outer = _current_default_options, **self.kwargs) # insert new scope at head of link
        return self
    # exiting with block: restore previous remembered defaults
    def __exit__(self, type, value, traceback):
        global _current_default_options
        _current_default_options = _current_default_options._outer # restore outer scope
        #self.new_scope._outer = None # (for good measure)

# options scope without limit to specific functions
def default_options(**kwargs):
    return _OptionsStack(None, **kwargs)

# options scope with limit to specific functions
# functions = function or list of functions
def default_options_for(functions, **kwargs):
    if not isinstance(functions, list):
        functions = [functions]
    return _OptionsStack(set(functions), **kwargs)

# return the up-to-date default option.
#def _get_current_default_options():
#    return _current_default_options
