import numpy as np
from sistemasLineales import SistemaLineal

def metodo_potencia(A, x, max_iter=1000, tol=1e-6):
    """
    Implementación del método de la potencia para encontrar el autovalor dominante y su autovector.
    
    Parámetros:
    A : numpy.ndarray
        Matriz cuadrada de la cual se desea encontrar el autovalor dominante.
    x : numpy.ndarray
        Vector inicial.
    max_iter : int
        Número máximo de iteraciones.
    tol : float
        Tolerancia para la convergencia.
    
    Retorna:
    lambda_dom : float
        Autovalor dominante.
    x : numpy.ndarray
        Autovector correspondiente al autovalor dominante.
    """
    x = x / x[0]
    lambda_dom = 0
    
    iter = 0
    for _ in range(max_iter):
        y = np.dot(A, x)
        lambda_dom_new = y[0]
        print('y', y)
        y = y / lambda_dom_new


        iter += 1
        if np.abs(np.linalg.norm(x - y)) < tol:
            break
        
        x = y
        lambda_dom = lambda_dom_new
    
    return lambda_dom, x, iter
    
def metodo_potencia_inversa(A, x=np.array([0]), max_iter=1000, tol=1e-10):
    n = A.shape[0]
    q = x.transpose().dot(A).dot(x) / x.transpose().dot(x)
    x = x / x[0]

    print('q ', q)
    
    iter = 0
    for _ in range(max_iter):
        x_old = x
        x = np.linalg.solve(A - q * np.eye(n), x)
        eigenvalue = x[0]
        x = x / eigenvalue

        
        iter+=1
        if np.linalg.norm(x - x_old) < tol:
            break
    
    eigenvalue = (1/eigenvalue) + q
    return eigenvalue, x, iter

# Ejemplo de uso
A = np.array([[4, 1], [2, 3]])
x = np.array([1, 2])
eigenvalue, eigenvector, iter = metodo_potencia(A, x)
print("Autovalor dominante:", eigenvalue)
print("Autovector correspondiente:", eigenvector)
print("Autovector de la matriz original:", np.linalg.eig(A))
eigenvalue, eigenvector, iter = metodo_potencia_inversa(A, x)
print("Autovalor dominante:", eigenvalue)
print("Autovector correspondiente:", eigenvector)

# Ejemplo de uso
A = np.array([[2, 1],
              [1, 3]])
x = np.array([1, 1])

autovalor, autovector,  iter = metodo_potencia(A, x)
print("Autovalor dominante:", autovalor)
print("Autovector correspondiente:", autovector)
print("Autovector de la matriz original:", np.linalg.eig(A))
eigenvalue, eigenvector, iter = metodo_potencia_inversa(A, x)
print("Autovalor dominante:", eigenvalue)
print("Autovector correspondiente:", eigenvector)

