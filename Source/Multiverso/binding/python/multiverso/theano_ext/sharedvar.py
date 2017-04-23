#!/usr/bin/env python
# coding:utf8

from theano.tensor.basic import TensorType, _tensor_py_operators
from theano.compile import SharedVariable
from theano.compile.sharedvalue import shared
from theano.gof import Variable, utils
import numpy
import multiverso as mv


class MVSharedVariable(object):
    '''MVSharedVariable is an wrapper of SharedVariable

    It will act same as SharedVariable. The only difference is a multiverso
    ArrayTable is addded to make it easier to sync values.
    '''
    def __init__(self, svobj):
        '''Constructor of the MVSharedVariable

        The constructor will create ArrayTableHandler and associate the shared
        variable with it. The initial value of ArrayTableHandler will be same
        as the value of SharedVariable. 
        *Notice*: Only the `init_value` from the master will be used!
        '''
        assert(isinstance(svobj, SharedVariable))
        self._svobj = svobj
        self._mv_array = mv.ArrayTableHandler(self._svobj.get_value().size,
                                              init_value=self._svobj.get_value().reshape((-1,)))

        mv.barrier()  # add barrier to make sure the initial values have token effect
        # _last_mv_data restore a copy of value. It will be used for calculate
        # the update for multiverso when calling mv_sync
        self._last_mv_data = self._mv_array.get().reshape(self._svobj.get_value().shape)
        self._svobj.set_value(self._last_mv_data, borrow=False)

    def mv_sync(self):
        ''' sync values with multiverso server

        mv_sync will add the delta of SharedVariable, which is usually the
        gradients in typical examples, to parameter server and then get the
        latest value in multiverso.
        '''
        # because multiverso always use add method to sync value, the delta
        # will be the difference of the current value of last synced value
        self._mv_array.add(self._svobj.get_value() - self._last_mv_data)

        self._svobj.set_value(self._mv_array.get().reshape(self._svobj.get_value().shape))
        self._last_mv_data = self._svobj.get_value(borrow=False)

    def __getstate__(self):
        '''This is for cPickle to store state.

        It is usually called when you want to dump the model to file with
        cPickle
        '''
        odict = self.__dict__.copy()  # copy the dict since we change it
        del odict['_mv_array']  # remove mv_array, because we can't pickle it
        return odict

    def __getattribute__(self, attr):
        '''This function make MVSharedVariable act same as SharedVariable'''
        if attr in ['_svobj', '_mv_array', '_last_mv_data']:
            # If get the attribute of self, use parent __getattribute__ to get
            # attribute from the object, otherwise it will fall into infinite
            # loop
            return object.__getattribute__(self, attr)
        elif attr in ['mv_sync', "__getstate__"]:
            # If call method of MVSharedVariable, then call the method directly
            # and bound the method to self object
            return getattr(MVSharedVariable, attr).__get__(self)
        else:
            # Otherwise I will get attribute from the wrapped object
            return getattr(self._svobj, attr)


def mv_shared(*args, **kwargs):
    '''mv_shared works same as `theano.shared`

    It calls `theano.shared` to create the SharedVariable and use
    MVSharedVariable to wrap it.
    '''
    var = shared(*args, **kwargs)
    mv_shared.shared_vars.append(MVSharedVariable(var))
    return var


mv_shared.shared_vars = []  # all shared_vars in multiverso will be recorded here


def sync_all_mv_shared_vars():
    '''Sync shared value created by `mv_shared` with multiverso

    It is often used when you are training model, and it will add the gradients
    (delta value) to the server and update the latest value from the server.
    Notice: It will **only** sync shared value created by `mv_shared`
    '''
    for sv in mv_shared.shared_vars:
        sv.mv_sync()
