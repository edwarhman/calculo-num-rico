import pytest
import numpy as np
from interpolacion import Puntos, Polinomio

tablasPuntos = [np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [9, 10],
])]


def expresion(x):
    return x**2


datos_construir_polinomio = [
    (np.array([1, 2]), '2*x + 1'),
    (np.array([0, 1]), 'x'),
    (np.array([1, 0]), '1'),
    (np.array([0, 0, 1]), 'x**2'),
]

datos_evaluar_polinomio = [
    (np.array([0, 0, 1]), 3, 9),
    (np.array([0, 0, 1]), 0, 0),
    (np.array([0, 0, 1]), 1, 1),
    (np.array([0, 0, 1]), 2, 4),
]


datos_construir_desde_exp = [
    (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), expresion)]


class TestPuntosInterpolacion:
    @pytest.mark.parametrize("tabla", tablasPuntos)
    def test_construccion_puntos(self, tabla):
        print(tabla)
        puntos = Puntos(tabla[:, 0], tabla[:, 1])
        assert np.array_equal(puntos.x, tabla[:, 0])
        assert np.array_equal(puntos.y, tabla[:, 1])

    @pytest.mark.parametrize("entrada, expresion", datos_construir_desde_exp)
    def test_construccion_puntos_desde_exp(self, entrada, expresion):
        print(entrada, expresion(entrada))
        puntos = Puntos.construirDesdeExpresion(entrada, expresion)
        assert np.array_equal(puntos.x, entrada)
        assert np.array_equal(puntos.y, expresion(entrada))

    @pytest.mark.parametrize("tabla", tablasPuntos)
    def test_construccion_puntos_desde_tabla(self, tabla):
        puntos = Puntos.construirDesdeTabla(tabla)
        assert np.array_equal(puntos.x, tabla[:, 0])
        assert np.array_equal(puntos.y, tabla[:, 1])


class TestPolinomioInterpolacion:
    @pytest.mark.parametrize(
        "coeficientes, expresion",
        datos_construir_polinomio
    )
    def test_construccion_polinomio(self, coeficientes, expresion):
        polinomio = Polinomio(coeficientes)
        assert str(polinomio) == expresion
        assert np.array_equal(polinomio.coeficientes, coeficientes)
        assert polinomio.grado == len(coeficientes) - 1

    @pytest.mark.parametrize(
        "coeficientes, entrada, salida",
        datos_evaluar_polinomio
    )
    def test_evaluar_polinomio(self, coeficientes, entrada, salida):
        polinomio = Polinomio(coeficientes)
        assert np.array_equal(polinomio.evaluar_polinomio(entrada), salida)
