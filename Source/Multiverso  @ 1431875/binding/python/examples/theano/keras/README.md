# Keras example

[addition_rnn_mv.py](./addition_rnn_mv.py) is adapted from
[a keras official example](https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py).


It will demonstrate how to use multiverso in keras.

For example, you can train it with two GPUs with such command.
```
mpirun -np 2 python addition_rnn_mv.py
```

It will reach `val_acc: 0.99+` much earlier than training with only one GPU.
