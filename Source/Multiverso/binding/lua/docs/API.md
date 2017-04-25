# Multiverso Torch Binding API

## init(sync)

Initialize mutliverso.

This should be called only once before training at the beginning of the whole
project.

If sync is `true`, a sync server will be created. Otherwise an async server
will be created.

If a sync server is created, you *must* make sure every process call
`add` and `get` in the same order and for the same times. Otherwise some
processes will be blocked. In sync server mode, all `get` method will
return *exactly the same results*.
If a async server is created, there won't be limitations like a sync
server. But we can't make sure `get` method will return the same results.
If you want to get the same results in async server mode, you should use
`barrier` and `get` with the argument `sync` set to `true` to sync the
processes.

## barrier()

Set a barrier for all workers to wait.

Workers will wait until all workers reach a specific barrier.

## shutdown()

Shutdown multiverso.

This should be called only once after finishing training at the end of the whole
project.

## num_workers()

Return the total number of workers.

## worker_id()

Return the id (zero-based index) for current worker.

## TableHandler

`TableHandler` is an interface to sync different kinds of values.

In most cases, you are supposed to sync models (for initialization) and
gradients (during training) so as to let multiverso help you manage the models
in distributed environments. Currently, two types of `TableHandler` are
supported, namely `ArrayTableHandler` and `MatrixTableHandler`.

### ArrayTableHandler

`ArrayTableHandler` is used to sync array-like (one-dimensional) value.

Although the model tends to be a matrix, when using `torch.nn` package we can
get the flattened parameters and gradients with
[module.getParameters()](https://github.com/torch/nn/blob/master/doc/module.md#flatparameters-flatgradparameters-getparameters).
So in most cases, we should use `ArrayTableHandler` instead of
`MatrixTableHandler` we will introduce soon.

#### ArrayTableHandler:new(size)

Create a `ArrayTableHandler` for syncing array-like (one-dimensional) value.

The `size` should be a `number` equal to the size of value we want to sync.

If init_value is nil, zeros will be used to initialize the table, otherwise
the table will be initialized as the init_value.
*Notice*: Only the init_value from the master will be used!

#### ArrayTableHandler:add(data, sync)

Add a array-like (one-dimensional) data to the server.

The `data` should be a `torch.Tensor` or Lua `table`. During training process,
the data should be the gradients (delta value). The size of `data` must be equal
to the size specified in initialization.

`sync` should be a boolean value. The default value is false. If `sync` is
`true`, this call will blocked by IO until the call finish.  Otherwise it will
return immediately

#### ArrayTableHandler:get()

Get the array-like (one-dimensional) value from the server.

The value we get will be a `torch.Tensor`. Usually, we are supposed to use
[Tensor:copy()](https://github.com/torch/torch7/blob/master/doc/tensor.md#self-copytensor)
to assign the value to desired destination.

### MatrixTableHandler

`MatrixTableHandler` is used to sync matrix-like (two-dimensional) value.

#### MatrixTableHandler:New(num_row, num_col, init_value)

Create a `MatrixTableHandler` for syncing matrix-like (two-dimensional) value.

The `num_row` should be the number of rows and the `num_col` should be the
number of columns. Both of them should be a `number` equal to the exact size of
value we want to sync.

If init_value is nil, zeros will be used to initialize the table, otherwise
the table will be initialized as the init_value.
*Notice*: if the init_value is different in different processes, the average of
them will be used.

#### MatrixTableHandler:add(data, row_ids, sync)

Add a matrix-like (two-dimensional) data to the server.

Same as the clarification in `ArrayTableHandler`, the `data` should be a
`torch.Tensor` or Lua `table` and we should pass the gradients (delta value) not
the exact value to it. The `row_ids` is an optional parameter and it should be
an array of 'row_id' numbers when specified. If specified, multiverso will only
update the value in specific rows and the size of `data` should be equal to the
size of value we want to update.

`sync` should be a boolean value. The default value is false. If `sync` is
`true`, this call will blocked by IO until the call finish.  Otherwise it will
return immediately

#### MatrixTableHandler:get(row_ids)

Get the matrix-like (two-dimensional) value from the server.

The `row_ids` is an optional parameter and the interface works the same way as
`ArrayTableHandler` when `row_ids` is not specified. But when we pass an array
of `row_id` numbers, we will only get the value form specific rows. In this way,
we can not do a `Tensor:copy()` but have to deal with the value manually.
