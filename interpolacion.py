# -*- coding: utf-8 -*-
"""practica_2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12i00NYfF23bXxG1Q8tsvrjUmSyivfV_m
"""

import numpy as np
from sympy  import symbols, plot, exp, cos, lambdify, latex, log
from sistemasLineales import resolverSistemaEliminacionGaussiana
import sys

print(sys.path)
xsym = symbols('x')

def construirPolinomio(coeficientes, grado, sym):
    poly = 0
    for i in range(grado, -1, -1):
        poly = poly + coeficientes[i] * (sym ** i)
    return poly

"""## Interpolaciòn Coeficientes ideterminados

---


"""

def interpolacionCoeficientesIdeterminados(x, y):
    X = np.vander(x, increasing=True)
    ai = resolverSistemaEliminacionGaussiana(X, y)
    poly = construirPolinomio(ai, len(ai)-1, xsym)
    return poly

"""## Polinomio interpolante de Lagrange"""

def polinomioInterpolanteDeLagrange(x, y):
    n = x.size
    sol = 0
    for i in range(n):
        l = 1
        for j in range(n):
            if j != i:
                mul = (xsym - x[j]) / (x[i] - x[j])
                l = l * mul
        sol = sol + y[i] * l
    return sol

"""## Diferencias Divididas"""

def diferenciasDivididas(x, y):
    n = x.size
    sol = np.copy(y)

    for i in range(1, n):
        sol[i:n] = (sol[i:n] - sol[i-1]) / (x[i:n] - x[i-1])
    return sol

"""## Polinomio Interpolante de Newton"""

def polinomioInterpolanteDeNewton(x, y):
    n = x.size
    sol = 0
    c = diferenciasDivididas(x, y)

    for i in range(n):
        mul = 1
        for j in range(i):
            mul = mul * (xsym - x[j])
        sol = sol + c[i] * mul
    return sol

"""## Ajuste minimos cuadrados"""

def ajusteMinimosCuadrados(x, y, m):
    if(len(x) != len(y)) :
        print("Los vectores deben tener la misma longitud")
        return
    n = len(x)
    # Inicializar matriz gradiente y vector
    matriz = np.zeros((m, m))
    vector = np.zeros(m)

    # Iterar para encontrar los m coeficientes
    for i in range(m):
        # Encontrar valores de la matriz
        for j in range(i, m):
          sumaMatriz = 0
          for k in range(n):
            sumaMatriz = sumaMatriz + x[k] ** (i + j)
          matriz[i,j] = sumaMatriz

        # Encontrar valores del vector
        sumaVector = 0
        for k in range(n):
          sumaVector = sumaVector + x[k] ** i * y[k]
        vector[i] = sumaVector

    # Llenar la matriz inferrior de la matriz copiando la transpuesta de la matriz y restando la diagonal principal
    matriz = matriz + matriz.transpose() - np.diag(matriz.diagonal())
    # asignamos a la primera celda de la matriz el numero de coeficientes que queremos encontrar
    matriz[0, 0] = n

    # Encontramos y devolvemos los coeficientes a resolviendo el sistema de ecuaciones lineales
    ai = resolverSistemaEliminacionGaussiana(matriz, vector)
    return ai
