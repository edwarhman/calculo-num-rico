import numpy as np

def resolverEcuacionBiseccion(f, a, b, tol, Iter = 100):
  c = a + (1/2)*(b - a)
  k = 0
  aa = np.empty((Iter,1))
  bb = np.empty((Iter,1))
  cc = np.empty((Iter,1))

  while b - a >= tol:
    aa[k,:] = a
    bb[k,:] = b
    cc[k,:] = c
    if np.sign(f(a)) != np.sign(f(c)): b = c
    if np.sign(f(a)) == np.sign(f(c)): a = c
    c = a + (1/2)*(b - a)
    k = k + 1

  return c, aa, bb, cc, k

def resolverEcuacionRegulaFalsi(f, a, b, tol, Iter = 100):
  k = 0
  numerador = a*f(b) - b*f(a)
  denominador = f(b) - f(a)
  c = numerador/denominador
  aa = np.empty((Iter+1,1))
  bb = np.empty((Iter+1,1))
  cc = np.empty((Iter+1,1))

  while np.abs(fun(c)) >= tol:
    if k > Iter:
      print('No se agoto el nómero de ieraciones:', k)
      break
    aa[k,:] = a
    bb[k,:] = b
    cc[k,:] = c
    if np.abs(fun(c))<tol: print('El cero de la función es:', c)
    if np.sign(fun(a))!= np.sign(fun(c)): b = c
    if np.sign(fun(a)) == np.sign(fun(c)): a = c
    numerador = a*fun(b) - b*fun(a)
    denominador = fun(b) - fun(a)
    c = numerador/denominador
    k = k + 1

  return c, aa, bb, cc, k

def resolverEcuacionSecante(fun, x0, x1, tol, Iter = 100):
  k = 0
  aa = np.empty((Iter,1))
  xx0 = np.empty((Iter,1))
  xx1 = np.empty((Iter,1))
  while np.abs(fun(x1))>= tol:
    if k > Iter: print('No se agoto el nómero de ieraciones:', k)
    aa[k,:] = a
    xx0[k,:] = x0
    xx1[k,:] = x1
    numerador = (x0 - x1)*fun(x1)
    denominador = fun(x0) - fun(x1)
    c = x1 - numerador/denominador
    x0 = x1
    x1 = c
    k = k + 1
  return x1, aa, xx0, xx1, k

def resolverEcuacionNewton(fun, x0, tol, MaxIter=100):
    k = 0
    xx = np.empty((MaxIter,1))

    while np.abs(fun(x0))>= tol:
        if k > MaxIter: print('No se agoto el nómero de ieraciones:', k)
        xx[k,:] = x0
        numerador = fun(x0)
        denominador = derifun(fun,x0)
        x1 = x0 - numerador/denominador
        x0 = x1
        k = k + 1
    return x0, xx, k

def derifun(fun,x):
    h = np.cbrt(np.finfo(float).eps)
    derifun = 0
    x_adelante = x + h
    x_atras = x - h
    derifun = (fun(x_adelante) - fun(x_atras))/(2*h)
    return derifun

def resolverEcuacionPuntoFijo(fun, x0, tol, MaxIter = 100):
  k = 0
  x1 = fun(x0)
  cc = np.empty((MaxIter,1))
  while abs(x0 - x1) >= tol:
    if k >= MaxIter:
      print('Se agoto el nómero de ieraciones:', k)
      break
    x0 = x1
    cc[k,:] = x1
    x1 = fun(x0)
    if k ==  MaxIter: print('Se alcanzo el máximo número de iteraciones')
    k  = k + 1
  print(k)
  return x1, cc, k

def grad(f,x):
    h = np.cbrt(np.finfo(float).eps)
    d = len(x)
    nabla = np.zeros(d)
    for k in range(d):
        xade = np.copy(x)
        xatr = np.copy(x)
        xade[k] += h
        xatr[k] -= h
        nabla[k] =(f(xade) - f(xatr))/(2*h)
    return nabla

def jacobiana(F, x):
  n = len(x)
  J = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      J[i,j] = grad(F[i],x)[j]
  return J

def resolverSistemaNoLinealNewton(F, x, x0, tol=1e-6, maxIter=1000):
  k = 0
  y = x0
  while np.linalg.norm(y) >= tol:
    if (k >= maxIter):
      print('Se alcanzo el máximo número de iteraciones')
      break
    J = jacobiana(F, x)
    F_e = np.array(list(map(lambda f: f(x), F)))
    y = np.linalg.solve(J, -F_e)
    x = x + y
    k = k + 1
  return x, k
