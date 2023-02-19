import numpy as np
from numpy import linalg as LA

# Divisions for real space mesh
S = np.array([20,25,30])

# Setup real space dimension of supercell, in Bohr radii
R = np.array([6.,6.,6.])
R = np.diag(R)

# Atomic positions
X = np.array([[0,0,0],[1.75,0,0]])

# Nuclear charge
Z = 1.0

# Form the M matrix
ms = np.arange(np.prod(S))
m1 = np.remainder(ms,S[0])
m2 = np.remainder(np.floor(ms/S[0]),S[1])
m3 = np.remainder(np.floor(ms/(S[0]*S[1])),S[2])

M = np.column_stack((m1,m2,m3))

# From the N matrix
n1 = m1-np.greater(m1,S[0]/2)*S[0]
n2 = m2-np.greater(m2,S[1]/2)*S[1]
n3 = m3-np.greater(m3,S[2]/2)*S[2]

N = np.column_stack((n1,n2,n3))

# Set up real space sample points
r = M @ LA.inv(np.diag(S)) @ R.T

# Set up reciprocal lattice vectors
G = 2*np.pi* N @ LA.inv(R)

# Square of the reciprocal vector, summed over the second dimension of the array
G2 = np.sum(G*G, axis=1, keepdims=True)

# Structure factor
Sf = np.sum(np.exp(-1j * G @ X.T), axis=1, keepdims=True)

# Calculate delta r from the center of the cell
cent = np.ones((np.prod(S),1))@np.sum(R,axis=1,keepdims=True).T*0.5
dr = np.sqrt(np.sum(np.square((r-cent)), axis=1, keepdims=True))
