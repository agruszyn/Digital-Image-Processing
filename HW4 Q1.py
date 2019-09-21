import dippykit as dip
import numpy as np

A = np.array([[90, 93, 156],[93, 156, 155],[156, 155, 155]])
I = np.array([[0.5,0.0],[0.0,0.5]])

Ahat = dip.resample(A, I, interpolation='bilinear')
print(Ahat)