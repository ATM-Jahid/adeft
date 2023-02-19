from config import *

#Define functions (need to move to separate files)
def fft3(dat,arg1, arg2):
    if arg2 == 1:
        return np.ndarray.reshape(np.fft.ifftn(np.ndarray.reshape(dat,arg1[2],arg1[1],arg1[0]))*np.prod(arg1),np.size(dat))
    else:
        return np.ndarray.reshape(np.fft.fftn(np.ndarray.reshape(dat,arg1[2],arg1[1],arg1[0])),np.size(dat))

def cI(arg1):
    return fft3(arg1,S,1)

def cJ(arg1):
    out=fft3(arg1,S,-1)/np.prod(S)
    out=out.reshape(-1,1)
    return out

def O(arg1):
    out = LA.det(R) * arg1
    return out

def L(arg1):
    out = -LA.det(R) * G2 * arg1
    return out

def Linv(arg1):
    np.seterr(divide='ignore', invalid='ignore')
    out = -1/LA.det(R) * np.reciprocal(G2) * arg1
    out[0] = 0
    return out
