from config import *
from functions import *

# Compute two gaussians
sigma1=0.25
g1 = (2*np.pi*sigma1**2)**(-3/2) * np.exp(-dr**2/2/sigma1**2) * Z

# Define the density from the structure factor
n = np.real(cI(cJ(g1)*Sf))

print(f'Normailization check, g1, {np.sum(g1)*np.linalg.det(R)/np.prod(S): .4f}')
print(f'Total Charge check, n, {np.sum(n)*np.linalg.det(R)/np.prod(S): .4f}')

# Calculate potential
phi = cI(Linv(-4*np.pi*O(cJ(n))))

# Change potential into column vector
phi=phi.reshape(-1,1)

# Calculate numerical and analytical couloumb energies
Unum = float(0.5 * np.real(cJ(phi).T.conj() @ O(cJ(n))))
Uself = Z**2 * 0.5 / np.sqrt(np.pi) / sigma1 * np.size(X, 0)
print(f'Unum-Uself: {Unum-Uself: .6f}')
