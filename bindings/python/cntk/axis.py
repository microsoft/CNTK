# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py
from .utils import typemap

__doc__='''
An axis object describes the axis of a variable and is used for specifying the axes parameters of certain functions such as reductions.
Besides the static axes corresponding to each of the axes of the variable's shape, variables of kind 'input' and any 
'output' variables dependent on an 'input' variable also have two additional dynamic axes whose dimensions are known only 
when the variable is bound to actual data during compute time (viz. sequence axis and batch axis denoting the axis along which
multiple sequences are batched).
'''

class Axis(cntk_py.Axis):
    '''
    Axis 
    
    '''

    def __init__(self, *args):
        this = _cntk_py.new_Axis(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    @typemap
    def is_ordered(self):
        '''
        Returns True if the axis is ordered; i.e. if there is an ordering between the dimensions along the axis.

        Returns:
            `bool`: True if this axis is ordered and False otherwise.
        '''
        return super(Axis, self).is_ordered()

    @typemap
    def is_static_axis(self):
        '''
        Returns True if the axis is of type static and False otherwise.

        Returns:
            `bool`: True if this axis is of type static and False otherwise.
        '''
        return super(Axis, self).is_static_axis()

    @typemap
    def name(self):
        '''
        Returns the name of this axis.

        Returns:
            `str`: the name of this axis.
        '''
        return super(Axis, self).name()

    @typemap
    def static_axis_index(self, checked=True):
        '''
        Returns the integer with which the static axis is defined. For example, 0 = first axis, 1 = second axis, etc.

        Args:
            checked (`bool`): if True then this function will throw an exception if the axis is not static.

        Returns:
            `int`: the numer with which the static axis is defined.
        '''
        kwargs=dict(locals()); del kwargs['self']; return super(Axis, self).static_axis_index(**kwargs)

