import numpy as np
import pytest
from sistemasLineales import SistemaLineal

sistemasValidos = [
    (
        np.array(
            [
                [1.0, 0.0, 1.0],
                [2.0, 3.0, -1.0],
                [1.0, 2.0, 1.0],
            ]
        ),
        np.array([2.0, 7.0, 6.0]),
    ),
    (
        np.array(
            [
                [1.0, 1.0, 0.0, 3.0],
                [2.0, 1.0, -1.0, 1.0],
                [3.0, -1.0, -1.0, 2.0],
                [-1.0, 2.0, 3.0, -1.0],
            ]
        ),
        np.array([4, 1, -3, 4]),
    ),
    (
        np.array(
            [
                [1.0, -1.0, 2.0, -1.0],
                [2.0, -2.0, 3.0, -3.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, -1.0, 4.0, 3.0],
            ]
        ),
        np.array([-8, -20, -2, 4]),
    ),
]

sistemasConSolucion = [
    (
        np.array(
            [
                [1.0, 0.0, 1.0],
                [2.0, 3.0, -1.0],
                [1.0, 2.0, 1.0],
            ]
        ),
        np.array([2.0, 7.0, 6.0]),
        np.array([1.0, 2.0, 1.0]),
    ),
    (
        np.array(
            [
                [1.0, 1.0, 0.0, 3.0],
                [2.0, 1.0, -1.0, 1.0],
                [3.0, -1.0, -1.0, 2.0],
                [-1.0, 2.0, 3.0, -1.0],
            ]
        ),
        np.array([4, 1, -3, 4]),
        np.array([-1.0, 2.0, 0.0, 1.0]),
    ),
    (
        np.array(
            [
                [1.0, -1.0, 2.0, -1.0],
                [2.0, -2.0, 3.0, -3.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, -1.0, 4.0, 3.0],
            ]
        ),
        np.array([-8, -20, -2, 4]),
        np.array([-7.0, 3.0, 2.0, 2.0]),
    ),
]

sistemasConSolucionLU = [
    (
        np.array(
            [
                [1.0, 0.0, 1.0],
                [2.0, 3.0, -1.0],
                [1.0, 2.0, 1.0],
            ]
        ),
        np.array([2.0, 7.0, 6.0]),
        np.array([1.0, 2.0, 1.0]),
    ),
    (
        np.array(
            [
                [1.0, 1.0, 0.0, 3.0],
                [2.0, 1.0, -1.0, 1.0],
                [3.0, -1.0, -1.0, 2.0],
                [-1.0, 2.0, 3.0, -1.0],
            ]
        ),
        np.array([4, 1, -3, 4]),
        np.array([-1.0, 2.0, 0.0, 1.0]),
    ),
]

sistemasConSolucionIterativa = [
    (
        np.array(
            [
                [5.0, -1.0, 3.0],
                [3.0, 6.0, 2.0],
                [2, 2.0, 4.0],
            ]
        ),
        np.array([-4.0, 11.0, 6.0]),
        np.array([-1.0, 2.0, 1.0]),
    ),
]

sistemasInvalidos = [
    (
        np.array(
            [
                [1.0, 1.0, 0.0, 3.0],
                [2.0, 1.0, -1.0, 1.0],
                [3.0, -1.0, -1.0, 2.0],
                [-1.0, 2.0, 3.0, -1.0],
                [-1.0, 2.0, 3.0, -1.0],
            ]
        ),
        np.array([4, 1, -3, 4]),
        "La matriz de coeficientes no es cuadrada.",
    ),
    (
        np.array(
            [
                [1.0, -1.0, 2.0, -1.0, 1.0],
                [2.0, -2.0, 3.0, -3.0, 2.0],
                [1.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, -1.0, 4.0, 3.0, 4.0],
            ]
        ),
        np.array([-8, -20, -2, 4]),
        "La matriz de coeficientes no es cuadrada.",
    ),
    (
        np.array(
            [
                [1.0, 1.0, 0.0, 3.0],
                [2.0, 1.0, -1.0, 1.0],
                [3.0, -1.0, -1.0, 2.0],
                [-1.0, 2.0, 3.0, -1.0],
            ]
        ),
        np.array([-8, -20, -2, 4, 12]),
        "El vector de términos independientes no es compatible con la matriz de coeficientes.",
    ),
    (
        np.array(
            [
                [1.0, 1.0, 0.0, 3.0],
                [2.0, 1.0, -1.0, 1.0],
                [3.0, -1.0, -1.0, 2.0],
                [-1.0, 2.0, 3.0, -1.0],
            ]
        ),
        np.array([-8, -20, -2]),
        "El vector de términos independientes no es compatible con la matriz de coeficientes.",
    ),
    ("A", "b", "Input variables must be numpy arrays."),
]


class TestSistemasLineales:
    @pytest.mark.parametrize("A, b", sistemasValidos)
    def test_object_construction(self, A, b):
        sistema = SistemaLineal(A, b)
        assert np.array_equal(sistema.A, A)
        assert np.array_equal(sistema.b, b)
        assert sistema.tamano == A.shape[0]

    @pytest.mark.parametrize("A, b, error_message", sistemasInvalidos)
    def test_object_construction_error(self, A, b, error_message):
        with pytest.raises(ValueError) as constructionError:
            SistemaLineal(A, b)
        assert error_message in str(constructionError.value)

    @pytest.mark.parametrize("A, b, expected_solution", sistemasConSolucion)
    def test_solver_eliminacion_gaussiana(self, A, b, expected_solution):
        sistema = SistemaLineal(A, b)
        solution = sistema.resolver_por_eliminacion_gaussiana()
        np.testing.assert_array_almost_equal(solution, expected_solution)

    @pytest.mark.parametrize("A, b, expected_solution", sistemasConSolucionLU)
    def test_solver_factorizacion_lu(self, A, b, expected_solution):
        sistema = SistemaLineal(A, b)
        solution = sistema.resolver_por_factorizacion_lu()
        np.testing.assert_array_almost_equal(solution, expected_solution)

    @pytest.mark.parametrize("A, b, expected_solution", sistemasConSolucion)
    def test_solver_factorizacion_palu(self, A, b, expected_solution):
        sistema = SistemaLineal(A, b)
        solution = sistema.resolver_por_factorizacion_palu()
        np.testing.assert_array_almost_equal(solution, expected_solution)

    @pytest.mark.parametrize(
        "A, b, expected_solution",
        sistemasConSolucionIterativa
    )
    def test_solver_metodo_jacobi(self, A, b, expected_solution):
        sistema = SistemaLineal(A, b)
        vector_inicial = np.ones(sistema.tamano)
        solution = sistema.resolver_por_metodo_jacobi(
            vector_inicial,
            iteraciones=43, tolerancia=1e-6)
        np.testing.assert_allclose(solution, expected_solution, rtol=1e-6)

    @pytest.mark.parametrize(
        "A, b, expected_solution",
        sistemasConSolucionIterativa
    )
    def test_solver_metodo_gaussSeidel(self, A, b, expected_solution):
        sistema = SistemaLineal(A, b)
        vector_inicial = np.ones(sistema.tamano)
        solution = sistema.resolver_por_metodo_gauss_seidel(
            vector_inicial,
            iteraciones=11, tolerancia=1e-6
        )
        np.testing.assert_allclose(solution, expected_solution, rtol=1e-6)

    @pytest.mark.parametrize(
        "A, b, expected_solution",
        sistemasConSolucionIterativa
    )
    def test_solver_metodo_sor(self, A, b, expected_solution):
        sistema = SistemaLineal(A, b)
        vector_inicial = np.ones(sistema.tamano)
        solution = sistema.resolver_por_metodo_sor(
            vector_inicial,
            iteraciones=11, tolerancia=1e-7, w=0.999
        )
        np.testing.assert_allclose(solution, expected_solution, rtol=1e-6)
