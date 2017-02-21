# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from .. import cntk_py

def crop(crop_type='center', crop_size=0, side_ratio=0.0, area_ratio=0.0, aspect_ratio=1.0, jitter_type='none'):
    '''
    Crop transform that can be used to pass to `map_features`

    Args:
        crop_type (str, default 'center'): 'center', 'randomside', 'randomarea',
          or 'multiview10'.  'randomside' and 'randomarea' are usually used during
          training, while 'center' and 'multiview10' are usually used during testing.
          Random cropping is a popular data augmentation technique used to improve
          generalization of the DNN.
        crop_size (`int`, default 0): crop size in pixels. Ignored if set to 0.
          When crop_size is non-zero, for example, crop_size=256, it means a cropping
          window of size 256x256 pixels will be taken. If one want to crop with
          non-square shapes, specify crop_size=256:224 will crop 256x224 (width x height)
          pixels. `When crop_size is specified, side_ratio, area_ratio and aspect_ratio
          will be ignored.`
        side_ratio (`float`, default 0.0): It specifies the ratio of final image
          side (width or height) with respect to the original image. Ignored if set
          to 0.0. Otherwise, must be set within `(0,1]`. For example, with an input
          image size of 640x480, side_ratio of 0.5 means we crop a square region
          (if aspect_ratio is 1.0) of the input image, whose width and height are
          equal to 0.5*min(640, 480) = 240. To enable scale jitter (a popular data
          augmentation technique), use colon-delimited values like side_ratio=0.5:0.75,
          which means the crop will have size between 240 (0.5*min(640, 480)) and 360
          (0.75*min(640, 480)).
        area_ratio (`float`, default 0.0): It specifies the area ratio of final image
          with respect to the original image. Ignored if set to 0.0. Otherwise, must be
          set within `(0,1]`. For example, for an input image size of 200x150 pixels,
          the area is 30,000. If area_ratio is 0.3333, we crop a square region (if
          aspect_ratio is 1.0) with width and height equal to sqrt(30,000*0.3333)=100.
          To enable scale jitter, use colon-delimited values such as area_ratio=0.3333:0.8,
          which means the crop will have size between 100 (sqrt(30,000*0.3333)) and
          155 (sqrt(30,000*0.8)).
        aspect_ratio (`float`, default 1.0): It specifies the aspect ratio (width/height
          or height/width) of the crop window. Must be set within `(0,1]`. For example,
          if due to size_ratio the crop size is 240x240, an aspect_ratio of 0.64 will
          change the window size to non-square: 192x300 or 300x192, each having 50%
          chance. Note the area of the crop window does not change. To enable aspect
          ratio jitter, use colon-delimited values such as aspect_ratio=0.64:1.0, which means
          the crop will have size between 192x300 (or euqally likely 300x192) and 240x240.
        jitter_type (str, default 'none'): crop scale jitter type, possible
          values are 'none' and 'uniratio'. 'uniratio' means uniform distributed jitter
          scale between the minimum and maximum ratio values.

    Returns:
        A dictionary-like object describing the crop transform
    '''
    return cntk_py.reader_crop(crop_type, crop_size, side_ratio,
        area_ratio, aspect_ratio, jitter_type)

def scale(width, height, channels, interpolations='linear', scale_mode="fill", pad_value=-1):
    '''
    Scale transform that can be used to pass to `map_features` for data augmentation.

    Args:
        width (int): width of the image in pixels
        height (int): height of the image in pixels
        channels (int): channels of the image
        interpolations (str, default 'linear'): possible values are
          'nearest', 'linear', 'cubic', and 'lanczos'
        scale_mode (str, default 'fill'): 'fill', 'crop' or 'pad'.
          'fill' - warp the image to the given target size.
          'crop' - resize the image's shorter side to the given target size and crop the overlap.
          'pad'  - resize the image's larger side to the given target size, center it and pad the rest
        pad_value (int, default -1): -1 or int value. The pad value used for the 'pad' mode.
         If set to -1 then the border will be replicated.

    Returns:
        A dictionary-like object describing the scale transform
    '''
    return cntk_py.reader_scale(width, height, channels,
            interpolations, scale_mode, pad_value)

def mean(filename):
    '''
    Mean transform that can be used to pass to `map_features` for data augmentation.

    Args:
        filename (str): file that stores the mean values for each pixel
         in OpenCV matrix XML format

    Returns:
        dict:
        A dictionary-like object describing the mean transform
    '''
    return cntk_py.reader_mean(filename)

def color(brightness_radius=0.0, contrast_radius=0.0, saturation_radius=0.0):
    '''
    Color transform that can be used to pass to `map_features` for data augmentation.

    Args:
        brightness_radius (float, default 0.0): Radius for brightness change. Must be
          set within [0.0, 1.0]. For example, assume brightness_radius = 0.2, a random
          number `x` is uniformly drawn from [-0.2, 0.2], and every pixel's value is
          added by `x*meanVal`, where meanVal is the mean of the image pixel intensity
          combining all color channels.
        contrast_radius (float, default 0.0): Radius for contrast change. Must be
          set within [0.0, 1.0]. For example, assume contrast_radius = 0.2, a random
          number `x` is uniformly drawn from [-0.2, 0.2], and every pixel's value is
          multiplied by `1+x`.
        saturation_radius (float, default 0.0): Radius for saturation change. Only for
          color images and must be set within [0.0, 1.0]. For example, assume
          saturation_radius = 0.2, a random number `x` is uniformly drawn from [-0.2, 0.2],
          and every pixel's saturation is multiplied by `1+x`.

    Returns:
        A dictionary-like object describing the mean transform
    '''
    return cntk_py.reader_color(brightness_radius, contrast_radius, saturation_radius)

#@staticmethod
#def intensity(intensity_stddev, intensity_file):
#    '''
#    Intensity transform that can be used to pass to `map_features` for data augmentation.
#    Intensity jittering based on PCA transform as described in original `AlexNet paper
#    <http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

#    Currently uses precomputed values from
#    https://github.com/facebook/fb.resnet.torch/blob/master/datasets/imagenet.lua

#    Args:
#        intensity_stddev (float): intensity standard deviation.
#        intensity_file (str): intensity file.
#    Returns:
#        dict describing the mean transform        '''
#    return dict(type='Intensity', intensityStdDev=intensity_stddev, intensityFile=intensity_file)
