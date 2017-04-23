# How to write python code with multiverso
1. You could start with the [test example](https://github.com/Microsoft/multiverso/blob/master/binding/python/multiverso/tests/test_multiverso.py) to learn the basic use of multiverso apis.
1. Try the [examples](https://github.com/Microsoft/multiverso/tree/master/binding/python/examples): The links of the original version are listed at the header part. All the modifications are commented with `# MULTIVERSO: XXX` and you can compare them with the original one.


Here is a typical usage of multiverso python binding.
```python
# import multiverso.
import multiverso as mv
# Init multiverso.
mv.init()
# Get total number of workers.
print mv.workers_num()
# Get the id for current worker.
print mv.worker_id()
# if the worker is master worker.
print mv.is_master_worker()

# Here is your code to create tables, train models and sync values.

# Shutdown multiverso
mv.shutdown()
```

Detailed api documents can be found in docstring of [api.py](https://github.com/Microsoft/multiverso/blob/master/binding/python/multiverso/api.py) and [tables.py](https://github.com/Microsoft/multiverso/blob/master/binding/python/multiverso/tables.py).

## About the sync server an async server

When initializing multiverso, you can create a sync server or an async server by setting the `sync` argument. `mv.init(sync=True)` will create a sync server. `mv.init(sync=False)` will create an async server. An async server will created by default.

If a sync server is created, you *must* make sure every process call `add` and `get` in the same order and for the same times. Otherwise some processes will be blocked. In sync server mode, all `get` method will return *exactly the same results*.

If a async server is created, there won't be limitations like a sync server. But we can't make sure `get` method will return the same results.  If you want to get the same results in async server mode, you should use `barrier` and `get` with the argument `sync` set to `True` to sync the processes.


## Model Initialization

Before actual training, we also need to make sure each worker has the same initial model for better training performance.

Multiverso use master strategy to initialize model. Only the init_value from the master will be used in the initial model on the server and then all workers fetch same initial models.

```python
# Create ArrayTableHandler for syncing parameters. In the constructor, Only
# the init_value from the master worker will be used in the initial model
tbh = mv.ArrayTableHandler(size, params)
# Wait for finishing the initializing phase.
mv.barrier()
# Get the initial model from the server.
params = tbh.get()
```
Similar strategies are already implemented in the constructors in `theano_ext.sharedvar` and `lasagne_ext.param_manager` during initialization.


## About the master worker
Some things should only be done in specific worker, such as validation, outputting the results and so on. So you can benefit from mv.is_master_worker() api to mark worker 0 as the master one to complete these tasks.
For example, if you want to make sure only one process will output the validation results, you can write similar code below.
```python
import multiverso as mv
# train your model
if mv.is_master_worker():
    # validate your model
    # print your validation results
```

Similar strategies are also applied in `theano_ext.sharedvar` and `lasagne_ext.param_manager` during initialization and already implemented in the constructors.



# How to use multiverso in theano
First, similarly, add `mv.init()`, `mv.shutdown()` and `mv.barrier()` mentioned above in your codebase.

In theano, parameters are usually stored in sharedVariables.

For example, sharedVariables can be created like this in a theano script.
```python
self.W = theano.shared(
    value=numpy.zeros(
        (n_in, n_out),
        dtype=theano.config.floatX
    ),
    name='W',
    borrow=True
)
```

If you want to use multiverso, you can modify them like this.
```python
from multiverso.theano_ext import sharedvar
W = sharedvar.mv_shared(
    value=numpy.zeros(
        (n_in, n_out),
        dtype=theano.config.floatX
    ),
    name='W',
    borrow=True
)

# build the model

# train the model

# When you are ready to add the delta of the variable to parameter
# server and sync the latest value, you can run this function
W.mv_sync()


# If you want to sync all variables created by `sharedvar.mv_shared`,
# you can use this function. It will add the gradients (delta value)
# to the server and update the latest value from the server.
sharedvar.sync_all_mv_shared_vars()
```

`mv_shared` is just a wrapper of `theano.shared`. It acts same as `theano.shared`, while making it more convenient to sync values.

`add` and `get` can also be used to sync parameters if you don't use shared variables.

Detailed api documents can be found in docstring of [sharedvar.py](https://github.com/Microsoft/multiverso/blob/master/binding/python/multiverso/theano_ext/sharedvar.py)


# How to use multiverso in lasagne
First, add `mv.init()`, `mv.shutdown()` and `mv.barrier()` mentioned above in your codebase.

Lasagne provides many functions to build models in theano. Multiverso python binding provides a manager to make managing and synchronizing the parameters in Lasagne more easily.

You can write code like this to manage your parameters.
A typical usage of managing the parameters is shown as below.
```python
from multiverso.theano_ext.lasagne_ext import param_manager

network = build_model()  # build_model is a function you implement to build model

# The LasagneParamManager will initialize the parameters and sync them with
# parameter server
lpm = param_manager.LasagneParamManager(network)

# Train the model

# When you are ready to add the delta of the variable in this model to the parameter
# server and get the latest value, you can run this function
lpm.sync_all_param()
```

Detailed api documents can be found in docstring of [param_manager.py](https://github.com/Microsoft/multiverso/blob/master/binding/python/multiverso/theano_ext/param_manager.py)


# How to use multiverso in Keras
First, add `mv.init()`, `mv.shutdown()` and `mv.barrier()` mentioned above in your codebase.

Keras provides many functions to build models. Multiverso python binding provides a callback function to make managing and synchronizing the parameters in Keras more easily.
This callback function will synchronize the parameters every mini-batch.

A typical usage of the callback function is shown as below.
```python
from multiverso.theano_ext.keras_ext.callbacks import MVCallback

model = Sequential()
# build and compile your model here

# Train the model
model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True,
          callbacks=[MVCallback(model)])  # The only difference is that you add callbacks here
```

The only difference from the normal keras program is that you add an extra callback function. This callback function will sync parameters every mini batch.

Detailed api documents can be found in docstring of [param_manager.py](https://github.com/Microsoft/multiverso/blob/master/binding/python/multiverso/theano_ext/param_manager.py) and [callbacks.py](https://github.com/Microsoft/multiverso/blob/master/binding/python/multiverso/theano_ext/keras_ext/callbacks.py)

# Run your multiverso program with 4 processes
Here is an example of running logistic regression with multi-process.
```
mpirun -np 4 python ./examples/theano/logistic_regression.py
```


# How to use multi-GPU in theano with multiverso
You need multiple GPUs on your server and have [CUDA backend](http://deeplearning.net/software/theano/tutorial/using_gpu.html#cuda) installed.

First, look through [this section](http://deeplearning.net/software/theano/install.html#using-the-gpu) to understand the configuration of which GPU will be used.

Second, run the program with multiverso in multiple processes.

Here is an example to make different processes use different GPUs.
In this example, the i-th worker will use the i-th GPU. You need to add code like this before `import theano`.
```python
import multiverso as mv
mv.init()
worker_id = mv.worker_id()
# NOTICE: To use multiple gpus, we need to set the environment before import theano.
if "THEANO_FLAGS" not in os.environ:
    os.environ["THEANO_FLAGS"] = 'floatX=float32,device=gpu%d,lib.cnmem=1' % worker_id

# import theano after this
```
