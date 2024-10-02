import pytest
import numpy as np
from autovalores_autovectores import metodo_potencia_inversa, metodo_potencia

datos_metodo_potencia = [
    # matrix, vector_inicial, autovalor_expected, autovector_expected
    (np.array([[2, 1], [1, 3]]), np.array([1, 1]), 100, 3.61803381, np.array([1., 1.6180338])),
    (np.array([[-2, -3], [6, 7]]), np.array([1, 1]), 6, 4.00440, np.array([1., -2.00037])),
    (np.array([[-4, 14, 0],
              [-5, 13, 0],
              [-1, 0, 2]]), np.array([1, 1, 1]), 12, 6.000837, np.array([1., 0.714316, -0.249895])),
]

tolerancia = 1e-6


class TestMetodoPotencia:
    @pytest.mark.parametrize("matrix, vector_inicial, iterations, autovalor_expected, autovector_expected", datos_metodo_potencia)
    def test_metodo_potencia(self, matrix, vector_inicial, iterations, autovalor_expected, autovector_expected):
        autovalor, autovector, iter = metodo_potencia(matrix, vector_inicial, max_iter=iterations)
        print('iterations ', iter)
        assert autovalor - autovalor_expected < tolerancia
        assert np.allclose(autovector, autovector_expected, atol=tolerancia)