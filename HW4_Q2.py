import dippykit as dip
import numpy as np


def transform(img, rows, cols, rot, size=None):
    s = np.sin(rot*180/np.pi)
    c = np.cos(rot*180/np.pi)
    r = np.array([[c, -s], [s, c]])
    m = np.array([[(1/rows), 0], [0, (1/cols)]])
    if size is None:
        img = dip.resample(img, m, interpolation='bicubic')
        return dip.resample(img, r, interpolation='bicubic')
    else:
        img = dip.resample(img, r, interpolation='bicubic')
        return dip.resample(img, m, crop = True, crop_size=size, interpolation='bicubic')


X = dip.im_read('images/cameraman.tif')
X = dip.im_to_float(X)
Xnew = transform(X, 2.5, 1.7, 27.5)
Xre = transform(Xnew, 1/2.5, 1/1.7, -27.5, np.shape(X))
Xdif = X-Xre

dip.subplot(2, 2, 1)
dip.xlabel('x')
dip.ylabel('y')
dip.title('Original Image')
dip.imshow(X, cmap='gray')

dip.subplot(2, 2, 2)
dip.xlabel('x')
dip.ylabel('y')
dip.title('transformed and rotated')
dip.imshow(Xnew, cmap='gray')

dip.subplot(2, 2, 3)
dip.xlabel('x')
dip.ylabel('y')
dip.title('transformed back')
dip.imshow(Xre, cmap='gray')

dip.subplot(2, 2, 4)
dip.xlabel('x')
dip.ylabel('y')
dip.title('difference between images')
dip.imshow(Xdif, cmap='gray')
print(X)
print(dip.PSNR(X, Xre, 1))

dip.show()