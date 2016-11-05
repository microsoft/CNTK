import numpy as np
import random as rnd
from PIL import Image
from scipy import ndimage

def compute_norm_mat(base_width, base_height): 
    # normalization matrix used in image pre-processing 
    x      = np.arange(base_width)
    y      = np.arange(base_height)
    X, Y   = np.meshgrid(x, y)
    X      = X.flatten()
    Y      = Y.flatten() 
    A      = np.array([X*0+1, X, Y]).T 
    A_pinv = np.linalg.pinv(A)
    return A, A_pinv

def preproc_img(img, A, A_pinv):
    # compute image histogram 
    img_flat = img.flatten()
    img_hist = np.bincount(img_flat, minlength = 256)

    # cumulative distribution function 
    cdf = img_hist.cumsum() 
    cdf = cdf * (2.0 / cdf[-1]) - 1.0 # normalize 

    # histogram equalization 
    img_eq = cdf[img_flat] 

    diff = img_eq - np.dot(A, np.dot(A_pinv, img_eq))

    # after plane fitting, the mean of diff is already 0 
    std = np.sqrt(np.dot(diff,diff)/diff.size)
    if std > 1e-6: 
        diff = diff/std
    return diff.reshape(img.shape)

def distort_img(img, out_width, out_height, max_shift, max_scale, max_angle, max_skew, flip=True): 
    shift_y = out_height*max_shift*rnd.uniform(-1.0,1.0)
    shift_x = out_width*max_shift*rnd.uniform(-1.0,1.0)

    # rotation angle 
    angle = max_angle*rnd.uniform(-1.0,1.0)

    #skew 
    sk_y = max_skew*rnd.uniform(-1.0, 1.0)
    sk_x = max_skew*rnd.uniform(-1.0, 1.0)

    # scale 
    scale_y = rnd.uniform(1.0, max_scale) 
    if rnd.choice([True, False]): 
        scale_y = 1.0/scale_y 
    scale_x = rnd.uniform(1.0, max_scale) 
    if rnd.choice([True, False]): 
        scale_x = 1.0/scale_x 
    T_im = crop_img(img, out_width, out_height, shift_x, shift_y, scale_x, scale_y, angle, sk_x, sk_y)
    if flip and rnd.choice([True, False]): 
        T_im = np.fliplr(T_im)
    return T_im

def crop_img(img, crop_width, crop_height, shift_x, shift_y, scale_x, scale_y, angle, skew_x, skew_y): 
    width, height = img.size
    center_x = width/2.0
    center_y = height/2.0    
    ctr_in = np.array((center_y, center_x))
    ctr_out = np.array((crop_height/2.0+shift_y, crop_width/2.0+shift_x))
    out_shape = (crop_height, crop_width)
    s_y = scale_y*(height-1)*1.0/(crop_height-1)
    s_x = scale_x*(width-1)*1.0/(crop_width-1)

    # rotation and scale 
    ang = angle*np.pi/180.0 
    transform = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    transform = transform.dot(np.array([[1.0, skew_y], [0.0, 1.0]]))
    transform = transform.dot(np.array([[1.0, 0.0], [skew_x, 1.0]]))
    transform = transform.dot(np.diag([s_y, s_x]))
    offset = ctr_in-ctr_out.dot(transform)

    # each point p in the output image is transformed to pT+s, where T is the matrix and s is the offset
    T_im = ndimage.interpolation.affine_transform(input = img, 
                                                  matrix = np.transpose(transform), 
                                                  offset = offset, 
                                                  output_shape = out_shape, 
                                                  order = 1,   # bilinear interpolation 
                                                  mode = 'reflect', 
                                                  prefilter = False)
    return T_im
