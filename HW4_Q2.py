import dippykit as dip
import numpy as np


# The Function for Q2 is defined here
# Takes in the image and parameters (rowise scaling, columnwise scaling, and rotation)
# The function is overloaded so that inputing the original size will do the inverse transform
def transform(img, rows, cols, rot, size=None):

    s = np.sin(rot*180/np.pi)                       # Sin(theta)
    c = np.cos(rot*180/np.pi)                       # Cos(theta)
    r = np.array([[c, -s], [s, c]])                 # Rotation Matrix
    m = np.array([[(1/rows), 0], [0, (1/cols)]])    # Sampling matrix

    # If doing a regular transformation, resize the image then rotate
    if size is None:
        img = dip.resample(img, m, interpolation='bicubic')
        return dip.resample(img, r, interpolation='bicubic')
    # If doing an inverse transformation, rotate image then resize
    else:
        img = dip.resample(img, r, interpolation='bicubic')
        return dip.resample(img, m, crop=True, crop_size=size, interpolation='bicubic')


# Read in the image
X = dip.im_read('images/cameraman.tif')
X = dip.im_to_float(X)

# Transform image
Xnew = transform(X, 2.5, 1.7, 27.5)

# Inverse transformation
Xre = transform(Xnew, 1/2.5, 1/1.7, -27.5, np.shape(X))

# Difference between original and new
Xdif = X-Xre

# Create a subplot, display the original image
dip.subplot(2, 2, 1)
dip.xlabel('x', fontsize=20)
dip.ylabel('y', fontsize=20)
dip.title('Original Image', fontsize=20)
dip.imshow(X, cmap='gray')

# Display transformed image
dip.subplot(2, 2, 2)
dip.xlabel('x', fontsize=20)
dip.ylabel('y', fontsize=20)
dip.title('b: transformed and rotated', fontsize=20)
dip.imshow(Xnew, cmap='gray')

# Display recreated image
dip.subplot(2, 2, 3)
dip.xlabel('x', fontsize=20)
dip.ylabel('y', fontsize=20)
dip.title('c: transformed back', fontsize=20)
dip.imshow(Xre, cmap='gray')

# Display the difference between images
dip.subplot(2, 2, 4)
dip.xlabel('x', fontsize=20)
dip.ylabel('y', fontsize=20)
dip.title('d: difference between images', fontsize=20)
dip.imshow(Xdif, cmap='gray')

# Calculate PSNR
print(dip.PSNR(X, Xre, 1))

# Show plots and finish
dip.show()
