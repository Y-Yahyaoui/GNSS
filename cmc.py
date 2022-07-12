
import numpy.matlib
import numpy as np

def app_coords(X, Y, Z, D):
    # X Y Z D are respectively arrays of 4 coordinates and distances of the first 4 satellites, their purpose is to calculate the approximate coords 
    B=np.array([[X[0]-X[1],[Y[0]-Y[1],[Z[0]-Z[1]],[X[1]-X[2],[Y[1]-Y[2],[Z[1]-Z[2]],[X[2]-X[3],[Y[2]-Y[3],[Z[2]-Z[3]]])
    C=np.array([D[1]**2-D[0]**2-(X[1]**2+Y[1]**2+Z[1]**2)+(X[0]**2+Y[0]**2+Z[0]**2)],[D[2]**2-D[1]**2-(X[2]**2+Y[2]**2+Z[2]**2)+(X[1]**2+Y[1]**2+Z[1]**2)],[D[3]**2-D[2]**2-(X[3]**2+Y[3]**2+Z[3]**2)+(X[2]**2+Y[2]**2+Z[2]**2)]))*0.5
    X0=np.dot(np.linalg.inv(B),C)
    return X0

def jacobienne(X, Y, Z, D, X0):
    # X Y Z D are respectively arrays of all coordinates and distances of the all the satellites, X0 are the app coords of the station
    A=np.zeros(len(X), 3, dtype=float)
    W=np.zeros(len(X), 1, dtype=float)
    for i in range(len(D)):
        E=sqrt((X0[0][0]-X[i])**2+(X0[1][0]-Y[i])**2+(X0[2][0]-Z[i])**2)
        A[i][0]=(X0[0][0]-X[i])/E
        A[i][1]=(X0[1][0]-Y[i])/E
        A[i][2]=(X0[2][0]-Z[i])/E
        W[i][0]=E-D[i]
    return A, W

def minimos_cuadrados(A, W, X0):
    N=np.dot(A.transpose(),A)
    U=np.dot(A.transpose(),W)
    X=np.dot(np.linalg.inv(N),U)
    Xcorr=X0-X
    return Xcorr

