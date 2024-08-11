import numpy as np
from sistemasLineales import resolverSistemaLU, resolverSistemaPALU, resolverSistemaEliminacionGaussiana

A = np.array([
    [1.,0.,1.],
    [2.,3.,-1.],
    [1.,2.,1.]
])

b = np.array([2.,7.,6.])

x = resolverSistemaLU(A, b)
print('LU')
print(x)
comprobacion = np.matmul(A, x)
print(comprobacion)

x = resolverSistemaPALU(A, b)
print('PALU')
print(x)
comprobacion = np.matmul(A, x)
print(comprobacion)

print('Eliminación Gaussiana')
x = resolverSistemaEliminacionGaussiana(A, b)
print(x)
comprobacion = np.matmul(A, x)
print(comprobacion)
