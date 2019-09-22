import dippykit as dip
import numpy as np

A = np.array([[90.0, 93.0, 156.0],[93.0, 156.0, 155.0],[156.0, 155.0, 155.0]])
I = np.array([[0.5,0.0],[0.0,0.5]])

Ahat = dip.resample(A, I, interpolation='bilinear')
print(Ahat)