# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py
from cntk.internal import typemap

class Axis(cntk_py.Axis):
    '''
    An axis object describes the axis of a variable and is used for specifying
    the axes parameters of certain functions such as reductions.  Besides the
    static axes corresponding to each of the axes of the variable's shape,
    variables of kind 'input' and any 'output' variables dependent on an
    'input' variable also have two additional dynamic axes whose dimensions are
    known only when the variable is bound to actual data during compute time
    (viz. sequence axis and batch axis denoting the axis along which multiple
    sequences are batched).

    Axis parameters can also be negative, which allows to refere axis starting
    from the last axis. Please be aware that Axis objects work in a
    column-major wise, as opposed to any other function in the library.
    '''

    def __init__(self, *args):
        super(Axis, self).__init__(*args)

    @property
    def is_ordered(self):
        '''
        Returns True if the axis is ordered; i.e. if there is an ordering between the dimensions along the axis.

        Returns:
            `bool`: True if this axis is ordered and False otherwise
        '''
        return super(Axis, self).is_ordered()

    @property
    def is_static_axis(self):
        '''
        Returns True if the axis is of type static and False otherwise

        Returns:
            bool: True if this axis is of type static and False otherwise
        '''
        return super(Axis, self).is_static_axis()

    @property
    def name(self):
        '''
        Returns the name of this axis.

        Returns:
            str: the name of this axis.
        '''
        return super(Axis, self).name()

    def static_axis_index(self, checked=True):
        '''
        Returns the integer with which the static axis is defined. For example, 0 = first axis, 1 = second axis, etc.

        Args:
            checked (bool): if True then this function will throw an exception if the axis is not static.

        Returns:
            `int`: the number with which the static axis is defined.
        '''
        return super(Axis, self).static_axis_index(checked)

    @staticmethod
    @typemap
    def default_dynamic_axis():
        '''
        Returns an Axis object representing the default dynamic axis

        Returns:
            :class:`Axis`: default dynamic axis
        '''
        return cntk_py.Axis.default_dynamic_axis()

    @staticmethod
    @typemap
    def default_batch_axis():
        '''
        Returns an Axis object representing the batch axis

        Returns:
            :class:`Axis`: default batch axis
        '''
        return cntk_py.Axis.default_batch_axis()

    @staticmethod
    @typemap
    def all_static_axes():
        '''
        Axis object representing all the static axes of an operand.

        Returns:
            :class:`Axis`: all static axes
        '''
        return cntk_py.Axis.all_static_axes()

    @staticmethod
    @typemap
    def all_axes():
        '''
        Axis object representing all the axes--static and dynamic--of an operand.

        Returns:
            :class:`Axis`: all axes
        '''
        return cntk_py.Axis.all_axes()

    @staticmethod
    @typemap
    def default_input_variable_dynamic_axes():
        '''
        Default dynamic axes of the input variable

        Returns:
            tuple of :class:`Axis`: instances
        '''
        return tuple(reversed(cntk_py.Axis.default_input_variable_dynamic_axes()))

    @staticmethod
    @typemap
    def unknown_dynamic_axes():
        '''
        Unknown dynamic axes

        Returns:
            tuple of :class:`Axis`: instances
        '''
        return tuple(reversed(cntk_py.Axis.unknown_dynamic_axes()))

    @staticmethod
    @typemap
    def new_unique_dynamic_axis(name):
        '''
        Creates an Axis object representing a new unique dynamic axis.

        Args:
            name (str): name of the dynmic axis

        Returns:
            :class:`Axis`: new unique dynamic axis
        '''
        return cntk_py.Axis.new_unique_dynamic_axis(name)

    @staticmethod
    @typemap
    def end_static_axis():
        '''
        DEPRECATED.

        Creates an Axis object representing a new leading static axis.

        Returns:
            :class:`Axis`: axis object representing a new leading static axis.
        '''
        import warnings
        warnings.warn('This will be removed in future versions. Please use '
                'Axis.new_leading_axis() instead.', DeprecationWarning)
        return cntk_py.Axis.end_static_axis()

    @staticmethod
    @typemap
    def new_leading_axis():
        '''
        Creates an Axis object representing a new leading static axis.

        Returns:
            :class:`Axis`: axis object representing a new leading static axis.
        '''
        return cntk_py.Axis.end_static_axis()
