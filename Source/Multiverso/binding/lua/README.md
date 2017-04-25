# Multiverso Torch/Lua Binding

## Introduction
Multiverso is a parameter server framework for distributed machine learning.
This package can enable parallel training of torch program over multiple machines and GPUs.

## Requirements
Build multiverso successfully by following the [README > build](https://github.com/Microsoft/multiverso/blob/master/README.md#build).

## Installation

**NOTE**: Before installation, you need to make sure have `libmultiverso.so`
built successfully according to [Requirements](#requirements).

```
make install
```
or
```
luarocks make
```

## Unit Tests
```
make test
```
or

```
luajit test.lua
```

## Documentation

- [Tutorial](https://github.com/Microsoft/multiverso/wiki/Integrate-multiverso-into-torch-project)
- [API](https://github.com/Microsoft/multiverso/wiki/Multiverso-Torch-Binding-API)
- [Benchmark](https://github.com/Microsoft/multiverso/wiki/Multiverso-Torch-Binding-Benchmark)
