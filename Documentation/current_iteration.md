# CNTK Current Iteration

## Efficient group convolution
The implementation of group convolution in CNTK has been updated. The updated implementation moves away from creating a sub-graph for group convolution (using slicing and splicing), and instead uses cuDNN7 and MKL2017 APIs directly. This improves the experience both in terms of performance and model size. 

As an example, for a single group convolution op with the following attributes:

- Input tensor (C, H, W) = (32, 128, 128)
- Number of output channels = 32 (channel multiplier is 1)
- Groups = 32 (depth wise convolution)
- Kernel size = (5, 5)

The comparison numbers for this single node are as follows:

| First Header  | GPU exec. time (in millisec., 1000 run avg.) | CPU exec. time (in millisec., 1000 run avg.) | Model Size (in KB, CNTK format)
| ------------- | ------------- | ------------- | ------------- |
| Old implementation  | 9.349  | 41.921  | 38  |
| New implementation  | 6.581  | 9.963  | 5  |
| Speedup/savings	Approx.  | 30%	Approx.  | 65-75%	Approx.  | 87% |

## Operators


## Bug fixes


## ONNX
### Updates
- Updated CNTK's ONNX BatchNormalization op export/import to latest spec.

### Bug or minor fixes:


## Misc

