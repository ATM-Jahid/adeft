from config import *
from functions import *
import scipy.linalg as sp
float_formatter = "{:.6f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# Define number of steepest decent iterations

Nit=600

# Compute potential

omega = 2
# ADD LINE TO CALCULATE POTENTIAL
V = 1/2 * omega**2 * dr**2

print(np.amax(V))
print(dr[15], V[15])

Vdual = cJdag(O(cJ(V)))

print(np.amax(Vdual))
print(dr[15], Vdual[15])

Ns = 4
np.random.seed(4)
W = np.random.rand(np.prod(S),Ns) + 1j*np.random.rand(np.prod(S),Ns)
E = getE(W, Vdual)
print(E)
