#!/usr/bin/env python
# coding:utf8

from keras.callbacks import Callback
from param_manager import KerasParamManager


class MVCallback(Callback):
    '''
    Please use MVCallback as a callback of keras model.fit function
    For e.g.
    ```
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[mvcallback(model, freq=1)])
    ```
    '''
    def __init__(self, model, freq=1):
        '''Initialize the MVCallback class

        The `model` should be the be a keras model
        The `freq` should be the update frequency of the parameters. For
        example, `freq=3` means update the parameters every 3 mini-batch.
        '''
        super(MVCallback, self).__init__()
        self.kpm = KerasParamManager(model)
        self.cur_n = 0
        if freq < 0:
            raise ValueError("Frequency must be an integer greater than 0.")
        self.freq = freq

    def on_batch_end(self, batch, logs={}):
        '''sync all parameters at the end of every batch'''
        self.cur_n = (self.cur_n + 1) % self.freq
        if self.cur_n % self.freq == 0:
            self.kpm.sync_all_param()
