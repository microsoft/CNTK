# XOR demo for multiverso.

The train example is referred from
https://github.com/torch/nn/blob/master/doc/training.md

There are two versions, where `xor.lua` is the raw version and
`xor-multiverso.lua` is the multiverso version.

Comments have been add to the every modification in `xor-multiverso.lua` that is
needed to make it run on multiverso.

## Run the raw version
```
make raw
```
or
```
th xor.lua
```

## Run the multiverso version
```
make multiverso
```
or
```
th xor-multiverso.lua
```
