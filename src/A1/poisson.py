from config import *
from functions import *

# Compute two gaussians
sigma1=0.75
#INSERT CODE HERE 

sigma2=0.5
#INSERT CODE HERE

# Define the density as the difference between the two gaussians
#INSERT CODE HERE

# print(f'Normailization check, g1, {np.sum(g1)*np.linalg.det(R)/np.prod(S): .4f}')
# print(f'Normailization check, g2, {np.sum(g2)*np.linalg.det(R)/np.prod(S): .4f}')
# print(f'Total Charge check, n, {np.sum(n)*np.linalg.det(R)/np.prod(S): .4f}')

#Calculate potential
# INSERT CODE HERE

## Change potential into column vector
# phi=phi.reshape(-1,1)

# #Calculate numerical and analytical couloumb energies
# Unum = float(0.5 * np.real(cJ(phi).T.conj() @ O(cJ(n))))
# Uanal=((1/sigma1+1/sigma2)*0.5-np.sqrt(2)/np.sqrt(np.power(sigma1,2)+np.power(sigma2,2)))/np.sqrt(np.pi)
# print(f'Calculated Hartree Energy = {Unum: .8f} Hartree')
# print(f'Exact Hartree Energy = {Uanal: .8f} Hartree')