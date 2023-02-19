from config import *
from functions import *

# Compute two gaussians
sigma1=0.75
g1 = (2*np.pi*sigma1**2)**(-3/2) * np.exp(-dr**2/2/sigma1**2)

sigma2=0.5
g2 = (2*np.pi*sigma2**2)**(-3/2) * np.exp(-dr**2/2/sigma2**2)

# Define the density as the difference between the two gaussians
n = g2 - g1

print(f'Normailization check, g1, {np.sum(g1)*np.linalg.det(R)/np.prod(S): .4f}')
print(f'Normailization check, g2, {np.sum(g2)*np.linalg.det(R)/np.prod(S): .4f}')
print(f'Total Charge check, n, {np.sum(n)*np.linalg.det(R)/np.prod(S): .4f}')

# Calculate potential
phi = cI(Linv(-4*np.pi*O(cJ(n))))

# Change potential into column vector
phi=phi.reshape(-1,1)

# Calculate numerical and analytical couloumb energies
Unum = float(0.5 * np.real(cJ(phi).T.conj() @ O(cJ(n))))
Uanal=((1/sigma1+1/sigma2)*0.5-np.sqrt(2)/np.sqrt(np.power(sigma1,2)+np.power(sigma2,2)))/np.sqrt(np.pi)
print(f'Calculated Hartree Energy = {Unum: .8f} Hartree')
print(f'Exact Hartree Energy = {Uanal: .8f} Hartree')
