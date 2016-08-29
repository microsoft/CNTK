This E2E test deals with 4x4 pixel grayscale images. The images are white except for 1 black and 1 gray pixel. 

The labels are the x and y positions of said pixels, normalized between -1 and 1, e.g.:
x_black	y_black	x_gray	y_gray

Some of the images only contain one gray/black pixel, thus the missing pixel is labeled as NAN.
Note that it is not possible to have an "all-NAN" label since the loss would not be computable.

Once the network is trained on 200 images, it is employed for another 50 images and writes the output predictions.