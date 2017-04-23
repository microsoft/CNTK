# Multiverso Torch Binding Benchmark

## Task Description

Perform CIFAR-10 classification with torch resnet implementation.

## Codebase

[Microsoft/fb.resnet.torch multiverso branch](https://github.com/Microsoft/fb.resnet.torch/tree/multiverso)

## Setup
Please follow [this guide](https://github.com/Microsoft/multiverso/wiki/Multiverso-Torch-Lua-Binding) to setup your environment.

## Hardware

- **Hosts** : 1
- **GPU** : Tesla K40m * 8
- **CPU** : Intel(R) Xeon(R) CPU E5-2680 v2 @ 2.80GHz
- **Memory** : 251GB

## Common settings

- depth 32
- nEpochs 164
- learningRate 0.1(epoch <= 80), 0.01(81 <= epoch <= 121), 0.001(121 <= epoch)

## Clarification for multiverso settings

- The train data is divided evenly to each worker.
- Master strategy is used to warm up the initial model.
- Workers sync after each batch and has a barrier after each epoch.

## Results

| Code Name | #Process(es) | #GPU(s) per Process | Use multiverso | Batch size | Initial learning rate | Seconds per epoch | Best Model |
| :-------: | :----------: | :-----------------: | :------------: | :--------: | :-------------------: | :---------------: | :--------: |
| 1P1G0M    | 1            | 1                   | 0              | 128        | 0.1                   | 55.57             | 92.435 %   |
| 1P8G0M    | 1            | 8                   | 0              | 128        | 0.1                   | 28.38             | 92.464 %   |
| 8P1G1M    | 8            | 1                   | 1              | 64         | 0.05                  | 11.37             | 92.449 %   |

![top1error_vs_epoch](https://raw.githubusercontent.com/Microsoft/multiverso/master/binding/lua/docs/imgs/top1error_vs_epoch.png)
![top1error_vs_runningtime](https://raw.githubusercontent.com/Microsoft/multiverso/master/binding/lua/docs/imgs/top1error_vs_runningtime.png)
