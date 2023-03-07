from config import *
import scipy.linalg as sp

#Define functions
def fft3(dat,arg1, arg2):
    if arg2 == 1:
        return np.ndarray.reshape(np.fft.ifftn(np.ndarray.reshape(dat,arg1[2],arg1[1],arg1[0]))*np.prod(arg1),np.size(dat))
    else:
        return np.ndarray.reshape(np.fft.fftn(np.ndarray.reshape(dat,arg1[2],arg1[1],arg1[0])),np.size(dat))

def cI(arg1):
    dim1=np.size(arg1,0)
    dim2=np.size(arg1,1)
    out=np.zeros((dim1,dim2),dtype='complex128')
    for x in range(0,dim2):
        out[:,x]=fft3(arg1[:,x],S,1)
    return out

def cJ(arg1):
    dim1=np.size(arg1,0)
    dim2=np.size(arg1,1)
    out=np.zeros((dim1,dim2),dtype='complex128')
    for x in range(0,dim2):
        out[:,x]=fft3(arg1[:,x],S,-1)/np.prod(S)
    return out

def cJdag(arg1):
    dim1=np.size(arg1,0)
    dim2=np.size(arg1,1)
    out=np.zeros((dim1,dim2),dtype='complex128')
    for x in range(0,dim2):
        out[:,x]=fft3(arg1[:,x],S,1)/np.prod(S)
    return out

def cIdag(arg1):
    dim1=np.size(arg1,0)
    dim2=np.size(arg1,1)
    out=np.zeros((dim1,dim2),dtype='complex128')
    for x in range(0,dim2):
        out[:,x]=fft3(arg1[:,x],S,-1)
    return out

def diagouter(arg1,arg2):
    out=np.sum(arg1*arg2.conj(),axis=1).reshape(-1,1)
    return out

def diagprod(arg1,arg2):
    out=arg1@np.ones((1,np.size(arg2,1)))*arg2 #BLAS2 style
    return out

def fdtest(arg1,Vdual):
    E0=getE(arg1,Vdual)
    g0=getgrad(arg1,Vdual)
    dW=np.random.rand(np.size(arg1,0),np.size(arg1,1))+1j*np.random.rand(np.size(arg1,0),np.size(arg1,1))
    for x in range(1,-10,-1):
        delta=10**x
        dE=2*np.real(np.trace(delta*g0.conj().T@dW))
        dE2=delta*g0.conj().T@dW
        print(delta)
        print('Actual dE = ', getE(arg1+delta*dW,Vdual)-E0)
        print('Projected dE = ', dE)
        print('Ratio = ', (getE(arg1+delta*dW,Vdual)-E0)/dE)
        #print(dE2)
    return
# Print iterations progress
def printProgressBar (iteration, total, energy, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        energy      - Requires  : Hartree (Float)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{energy:0.6f} {prefix} |{bar}| {percent}% {suffix}', end = '')
    # Print New Line on Complete
    if iteration == total:
        print()

#### Above are provided as is ###
def O(arg1):
    out = LA.det(R) * arg1
    return out

def Linv(arg1):
    #Extend your code for Linv using cI as a model to handle the multidimensional W array
    dim1=np.size(arg1,0)
    dim2=np.size(arg1,1)
    out=np.zeros((dim1,dim2),dtype='complex128')
    np.seterr(divide='ignore', invalid='ignore')
    for x in range(0, dim2):
        out[:,x] = -1/LA.det(R) * np.reciprocal(G2[:,0]) * arg1[:,x]
        out[0,x] = 0
    return out

def L(arg1):
    #Extend your code for L using cI as a model to handle the multidimensional W array
    dim1=np.size(arg1,0)
    dim2=np.size(arg1,1)
    out=np.zeros((dim1,dim2),dtype='complex128')
    for x in range(0, dim2):
        out[:,x] = -LA.det(R) * G2[:,0] * arg1[:,x]
    return out

def getE(arg1,Vdual):
    # Build this function based on the description in the assignment.
    # Okay to take real part after you confirm that only roundoff remains in imaginary component
    U = arg1.conj().T @ O(arg1)
    n = diagouter(cI(arg1) @ LA.inv(U), cI(arg1))
    out = -0.5 * np.trace(arg1.conj().T @ L(arg1) @ LA.inv(U)) + Vdual.conj().T @ n
    return np.real(out.item())

def H(arg1,Vdual):
    # Build H per instructions in the assignment
    out = -0.5 * L(arg1) + cIdag(diagprod(Vdual, cI(arg1)))
    return out

def getgrad(arg1,Vdual):
    #Build getgrad based on instructions in the assignment
    cHW = H(arg1,Vdual)
    cOW = O(arg1)
    cWd = arg1.conj().T
    cUi = LA.inv(cWd @ cOW)
    out = (cHW - (cOW @ cUi) @ (cWd @ cHW)) @ cUi
    return out

def sd(arg1,cnt,Vdual):
    alpha=0.00003
    old = arg1
    for i in range(cnt):
        new = old - alpha * getgrad(old, Vdual)
        printProgressBar(i, cnt, getE(new, Vdual))
        old = new

    #Build your steepest descent algorithm. It should return the optimized W as out
    return new

def getPsi(arg1,Vdual):
    U = arg1.conj().T @ O(arg1) # ADD Your code
    Y = arg1 @ sp.inv(sp.sqrtm(U)) # Add your code, see above equations
    mu = Y.conj().T @ H(Y, Vdual) # Add your code, see above equations
    epsilon,D=LA.eig(mu) # This line does not need changing
    epsilon=np.real(epsilon) # Does not need change, removing numerical round off in imaginary component
    Psi = Y @ D # Add your code, see above equations
    return Psi,epsilon # Does not need to be changed


##### For DFT part

def excVWN(arg1):
    # Add references
    X1 = 0.75*(3.0/(2.0*np.pi))**(2.0/3.0)
    A = 0.0310907
    x0 = -0.10498
    b = 3.72744
    c = 12.9352
    Q = np.sqrt(4.0*c-b**2)
    X0 = x0**2 + b*x0 + c

    rs= (4*np.pi/3.0*arg1)**(-1/3)

    x = np.sqrt(rs)
    X = x*x+b*x+c

    out = -X1/rs + A*(np.log(x*x/X)+2*b/Q*np.arctan(Q/(2*x+b))-(b*x0)/X0*(np.log((x-x0)*(x-x0)/X)+2*(2*x0+b)/Q*np.arctan(Q/(2*x+b))))
    return out

def excpVWN(arg1):
    # Add references
    X1 = 0.75*(3.0/(2.0*np.pi))**(2.0/3.0)
    A = 0.0310907
    x0 = -0.10498
    b = 3.72744
    c = 12.9352
    Q = np.sqrt(4.0*c-b**2)
    X0 = x0**2 + b*x0 + c

    rs= (4*np.pi/3.0*arg1)**(-1/3)

    x = np.sqrt(rs)
    X = x*x+b*x+c

    dx = 0.5/x

    out = dx*(2*X1/(rs*x)+A*(2/x-(2*x+b)/X-4*b/(Q*Q+(2*x+b)*(2*x+b))-(b*x0)/X0*(2/(x-x0)-(2*x+b)/X-4*(2*x0+b)/(Q*Q+(2*x+b)*(2*x+b)))))
    out = (-rs/(3*arg1))*out
    return out

