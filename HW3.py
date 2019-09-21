import numpy as np
import math
import dippykit as dip
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from scipy.interpolate import interp2d

# Read in the image and convert it to a normalized float range with im_to_float
X = dip.im_read('images/cameraman.tif')
X = dip.im_to_float(X)
# Scale up the image at the beginning. Remember, dip.im_to_float() converted
# the range of X from 0~255 to 0.0~1.0. We have to multiply by 255 to have
# the range be 0.0~255.0. Don't forget to scale it back down before using
# dip.float_to_im(), which will convert the image back to an integer type.
# X *= 255
#
# # Add 75 to each pixel value and save the resulting image as Y
# Y = X + 75
#
# # Save the image (don't forget to scale down Y!)
# dip.im_write(dip.float_to_im(Y/255), 'cameraman_add.tif')
#
#
# # Square all values of X and store the result to Z
# Z = X ** 2
#
# # Save the image (don't forget to scale down Z!)
# dip.im_write(dip.float_to_im(Z/255), 'cameraman_square.tif')
fit = np.shape(X)
#dip.show()########################333
rotate = np.array([[0, 1],[1, 0]])
M = []
M.append(np.array([[0, 10], [10, 0]]))
M.append(np.array([[0, 1], [10, 0]]))
M.append(np.array([[0, 10], [1, 0]]))
M.append(np.array([[1, 1], [1, 2]]))
M.append(np.array([[3, 1], [1, 2]]))
L = []
for i in M:
    L.append(np.linalg.inv(i))
Xd = []
for i in range(len(M)):
    Xd.append(dip.resample(X,M[i]))
Xu = []
for i in range(len(M)):
    Xu.append(dip.resample(Xd[i],L[i], crop = True, crop_size = fit, interpolation ='nearest'))

dip.subplot(3, 2, 1)
#dip.surf(X, Y, Z, cmap='summer')
dip.xlabel('x')
dip.ylabel('y')
dip.title('Original Image')
dip.imshow(X, cmap = 'gray')
dip.subplot(3, 2, 2)
#dip.surf(X, Y, Z, cmap='summer')
dip.xlabel('x')
dip.ylabel('y')
dip.title('i')
dip.imshow(Xd[0], cmap = 'gray')
dip.subplot(3, 2, 3)
dip.xlabel('x')
dip.ylabel('y')
dip.title('ii')
dip.imshow(Xd[1], cmap = 'gray')
dip.subplot(3, 2, 4)
dip.xlabel('x')
dip.ylabel('y')
dip.title('iii')
dip.imshow(Xd[2], cmap = 'gray')
dip.subplot(3, 2, 5)
dip.xlabel('x')
dip.ylabel('y')
dip.title('iv')
dip.imshow(Xd[3], cmap = 'gray')
dip.subplot(3, 2, 6)
dip.xlabel('x')
dip.ylabel('y')
dip.title('v')
dip.imshow(Xd[4], cmap = 'gray')
dip.show()
################
dip.subplot(3, 2, 1)
#dip.surf(X, Y, Z, cmap='summer')
dip.xlabel('x')
dip.ylabel('y')
dip.title('Original Image')
dip.imshow(X, cmap = 'gray')
dip.subplot(3, 2, 2)
#dip.surf(X, Y, Z, cmap='summer')
dip.xlabel('x')
dip.ylabel('y')
dip.title('i')
dip.imshow(Xu[0], cmap = 'gray')
dip.subplot(3, 2, 3)
dip.xlabel('x')
dip.ylabel('y')
dip.title('ii')
dip.imshow(Xu[1], cmap = 'gray')
dip.subplot(3, 2, 4)
dip.xlabel('x')
dip.ylabel('y')
dip.title('iii')
dip.imshow(Xu[2], cmap = 'gray')
dip.subplot(3, 2, 5)
dip.xlabel('x')
dip.ylabel('y')
dip.title('iv')
dip.imshow(Xu[3], cmap = 'gray')
dip.subplot(3, 2, 6)
dip.xlabel('x')
dip.ylabel('y')
dip.title('v')
dip.imshow(Xu[4], cmap = 'gray')
dip.show()

for i in Xu:
    print(dip.PSNR(X,i))
    Xr = dip.resample(X,rotate)
    fX = dip.fftshift(dip.fft2(X))
    fX = np.log(np.abs(fX))

Rspectrum = dip.fftshift(dip.fft2(Xr))
Rspectrum = np.log(np.abs(Rspectrum))
dip.subplot(2, 2, 1)
dip.xlabel('x')
dip.ylabel('y')
dip.title('original')
dip.imshow(X, cmap = 'gray')
dip.subplot(2, 2, 2)
dip.xlabel('x')
dip.ylabel('y')
dip.title('fourier original')
dip.imshow(fX, cmap = 'gray')
dip.subplot(2, 2, 3)
dip.xlabel('x')
dip.ylabel('y')
dip.title('rotated')
dip.imshow(Xr, cmap = 'gray')
dip.subplot(2, 2, 4)
dip.xlabel('x')
dip.ylabel('y')
dip.title('fourier rotated')
dip.imshow(Rspectrum, cmap = 'gray')
dip.im_write(dip.float_to_im(fX), 'cameraman_new.tif')
dip.show()