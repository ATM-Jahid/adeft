from config import *
from functions import *
import scipy.linalg as sp
float_formatter = "{:.6f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# Define number of steepest decent iterations

Nit=500

# Compute potential

omega = 2
# ADD LINE TO CALCULATE POTENTIAL
V = 1/2 * omega**2 * dr**2

Vdual = cJdag(O(cJ(V)))

Ns = 4
W = np.random.rand(np.prod(S),Ns) + 1j*np.random.rand(np.prod(S),Ns)
W = W @ sp.inv(sp.sqrtm(W.conj().T @ O(W)))

W = sd(W, Nit, Vdual)

Psi, epsilon = getPsi(W, Vdual)

print(f'Total energy: {getE(W, Vdual)}')
print(f'Eigenvalues: {epsilon}')


##################
# Debugging prints
##################

#fdtest(W, Vdual)

#print(np.amax(V))
#print(dr[15], V[15])

#print(np.amax(Vdual))
#print(dr[15], Vdual[15])

#E = getE(W, Vdual)
#print(E)

#np.random.seed(2021)
#a = np.random.rand(np.prod(S),1) + 1j * np.random.rand(np.prod(S),1)
#b = np.random.rand(np.prod(S),1) + 1j * np.random.rand(np.prod(S),1)
#
#a = a.reshape(-1,1)
#b = b.reshape(-1,1)
#
#c = np.conjugate(a.conj().T @ H(b, Vdual))
#d = b.conj().T @ H(a, Vdual)
#
#print(c)
#print(d)

#print('\n\n')
#print(np.real(Psi.conj().T @ O(Psi)))
#print(np.real(Psi.conj().T @ H(Psi, Vdual)))
#print(epsilon)
