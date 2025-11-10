import numpy as np
import scipy.integrate
from sympy import Matrix, dsolve, symbols, Function, lambdify, Eq, sympify, ImmutableMatrix, exp, zeros
from copy import copy

"""
This file is part of pySpec
    Copyright (C) 2024  Markus Bauer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    pySpec is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


class KineticModel:
    """
    This class serves to calculate kinetic models based on an input k-matrix representing the flow of chemical reactions.
    For example, the kinetic Matrix for Enzyme Kinetics à
    [S] + [E] <-> [ES] -> [P] + [E],
    which is written as::

        matrix = (
            "[[-k0 * f1(t),           0,     +k1, 0],"  # [S]
            " [          0, -k0 * f0(t), +k1 +k2, 0],"  # [E]
            " [ k0 * f1(t),           0, -k1 -k2, 0],"  # [ES]
            " [ 0,                    0,     +k2, 0]]"  # [P]
        )

    is used together with starting concentrations to prepare a set of anonymous functions. These, when populated with
    explicit k-values and a temporal profile, will calculate the concentrations for each provided timestep for each
    species using the method `func`:: calculate_concentrations, eg::

        start_conc = (1, 1, 0, 0)
        model = KineticModel(matrix, (1, 1, 0, 0))
        t = np.arange(1, 100)
        k = [1/10, 1/10, 1/100]

        c = model.calculate_concentrations(t, k)

    This class must be used for all global fitting routines and the SpecSynthesizer.

    Solving the kinetic differential equations symbolically can take a long time (the property _inversion_method has a
    large influence on this, which is why GE, QR, DM and DMNC are marked as excluded) or no analytical solution may
    exist, therefore purely numerical solutions can also be generated using the flag solve_symbol=False. This speeds up
    the initial model generation drastically, but it slows down evaluation at later points.
    """


    _solver = 'LSODA'  # 'RK45'
    _inversion_method = 'ADJ' # 'ADJ', 'LDL', 'LU', EXCLUDED: GE, QR, DM, DMNC

    def __init__(self,
                 matrix:       str = None,
                 start_conc:   list[float] or tuple[float] = None,
                 solve_symbol: bool = True):
        """
        :param matrix: String representation of the (square) k-Matrix. Concentrations must be denoted as f{i}(t),
                       where {i} is a zero-based index for each species.
        :param start_conc: A list or tuple of floats representing the initial concentrations of the species.
        :param solve_symbol: Flag indicating whether to attempt symbolic solution of differential equations.
                             If False, a numerical solver is used instead.
        """
        self.k_matrix = Matrix(sympify(matrix))
        self.start_conc = start_conc
        self._symbol_solve = solve_symbol

        self._test_input()

        self.function, self.parameter, self.func = self._create_model()

    def calculate_concentrations(self, times, k_list):
        """
        Calculate the concentrations of species at given times using either symbolic or numerical methods.

        :param times: Array of time points at which to evaluate the concentrations.
        :param k_list: List of rate constants to be used in the calculations.
        :return: Array of concentrations at the specified times.
        """

        if self._symbol_solve:
            return np.array(self.function(times, *k_list))
        else:
            res = scipy.integrate.solve_ivp(self.function,
                                             (0, times.max()),
                                             self.start_conc,
                                             t_eval=times,
                                             args=k_list,
                                             method=self._solver,
                                             dense_output=False)
            return res.y

    def _test_input(self):
        """
        Check that the k-matrix is square and that its dimensions match the number of species in the start_conc list.
        """

        assert self.k_matrix.is_square
        assert self.k_matrix.shape[0] == len(self.start_conc)

    def _create_model(self):
        """
        Create the kinetic model (either symbolic or numerical) based on the k-matrix.

        :return: A tuple of (lambda function to evaluate the system, list of parameters, symbolic equations)
        """
        t = symbols('t')
        sym = {t}

        sym.update(self.k_matrix.free_symbols)

        # sort ki and t; t will end up as last position
        arguments = sorted([x.name for x in sym])
        # copy list and get rid of t to obtain list of parameter useful for later optimisation algorithm
        parameter = copy(arguments)
        parameter.pop()
        print(parameter)
        # put t in first position in arguments for the lambda creation
        arguments.insert(0, arguments.pop())

        # Unimolecular approximation is possible, if no time dependency is found in the k-matrix.
        if t not in self.k_matrix.free_symbols and self._symbol_solve:
            # This method will result in lin. algebraic solution and can be directly converted to the evaluating lambda
            func = self._matrix_method(self.k_matrix, self.start_conc)
        else:
            # Else the power of sympy.dsolve is leveraged or a numerical approach is prepared.
            func, self._symbol_solve = self._dsolve_method(self.k_matrix, self.start_conc, self._symbol_solve)

        if not self._symbol_solve:
            # insert concentration placeholder symbols as list in second position after t for scipy.integrate.solve_ivp
            arguments.insert(1, [f'f{i}' for i, _ in enumerate(func)])

        return lambdify(arguments, func, 'numpy'), parameter, func

    def _matrix_method(self, k_matrix, start_conc):
        """
        Solve the system using an eigenvalue/eigenvector approach for unimolecular reactions
        (no time dependence in k-matrix).

        :param k_matrix: The k-matrix (reaction rate matrix).
        :param start_conc: List of initial concentrations.
        :return: List of concentrations as functions of time.
        """
        t = symbols('t')

        k_eigen = k_matrix.eigenvects()

        eigen_values = []
        eigen_vectors = Matrix()
        for res in k_eigen:
            for ev in res[-1]:
                eigen_values.append(res[0])
                eigen_vectors = eigen_vectors.col_insert(len(eigen_values), ev)

        eigen_vectors = ImmutableMatrix(eigen_vectors)

        a = eigen_vectors.inv(method=self._inversion_method) * Matrix(start_conc)

        ct = zeros(len(start_conc), 1)
        for i, _ in enumerate(start_conc):
            ct += a[i] * eigen_vectors.col(i) * exp(eigen_values[i] * t)

        return [x[0] for x in ct.tolist()]

    @staticmethod
    def _dsolve_method(k_matrix, start_conc, symbol_solve):
        """
        Solve the system using sympy's dsolve method (symbolic solution) or fallback to numerical method.

        :param k_matrix: The k-matrix (reaction rate matrix).
        :param start_conc: List of initial concentrations.
        :param symbol_solve: Flag indicating whether to try symbolic solving.
        :return: Tuple of (list of symbolic ODE solutions or numerical equations, success flag).
        """
        t = symbols('t')

        functions = [Function(f"f{i}")(t) for i, _ in enumerate(start_conc)]

        # Matrix Definitions
        c_matrix = Matrix(functions)

        ics = {}
        for func, conc in zip(functions, start_conc):
            ics[func.subs(t, 0)] = conc

        # Create Equation Matrix
        eq_matrix = k_matrix * c_matrix

        # Convert Matrices to lists
        eq = [Eq(lhs, rhs) for lhs, rhs in zip(c_matrix.diff(t), eq_matrix)]
        func = [x for x in c_matrix]

        # Solve
        if symbol_solve:
            try:
                print('Trying to solve the equation symbolically! This might take some time...')
                # Try first to symbolically solve the system of differential equations
                # If it is possible, return a list of the solution functions and a flag to signal success of the
                # symbolical evaluation.
                return [f.rhs for f in dsolve(eq, func=func, ics=ics)], True
            except NotImplementedError:
                print("ODE not symbolically solvable. Falling back on numerical solver!")
                pass

        # If the ODE is not symbolically solvable or numerical solving is selected, substitute all fi(t) functions with
        # fi symbols, append all the right hand sides of the unsolved differential equations to a list and return that
        # list, as well as a flag to signal failure of symbolical evaluation.
        deq = []
        for f in eq:
            deq.append(f.rhs)
            for i, fun in enumerate(functions):
                deq[-1] = deq[-1].subs(fun, f'f{i}')

        return deq, False

    @classmethod
    def create_from_N_matrix(cls, matrix):
        ...

    @classmethod
    def decay_associated_model(cls, component_amount, solve_symbol: bool = True):
        """
        Create a model for a decay process with a given number of components. Assumes that each component decays
        to the next component in a chain.

        :param component_amount: Number of components (species).
        :param solve_symbol: Flag indicating whether to try symbolic solving.
        :return: A KineticModel instance with a decay-associated k-matrix and default starting concentrations.
        """

        matrix = "["

        k = 0
        for i in range(component_amount):
            matrix += "["
            for j in range(component_amount):
                if i == j:
                    matrix += f"-k{k}"
                    k += 1
                else:
                    matrix += "0"

                if not j == component_amount-1:
                    matrix += ','

            matrix += ']'
            if not i == component_amount-1:
                matrix += ',\n'

        matrix += "]"

        print(matrix)
        return cls(matrix, start_conc=[1 for _ in range(component_amount)], solve_symbol=solve_symbol)

    @classmethod
    def evolution_associated_model(cls, component_amount, solve_symbol: bool = True):
        """
        Create a model for a decay process with a given number of components. Assumes that each component decays
        to the next component in a chain.

        :param component_amount: Number of components (species).
        :param solve_symbol: Flag indicating whether to try symbolic solving.
        :return: A KineticModel instance with a decay-associated k-matrix and default starting concentrations.
        """

        matrix = "["

        k = 0
        for i in range(component_amount):
            matrix += "["
            for j in range(component_amount):
                if (k - 1) == j:
                    matrix += f"k{k-1}"
                elif i == j and (i != component_amount - 1):
                    matrix += f"-k{k}"
                    k += 1
                else:
                    matrix += "0"

                if not j == component_amount-1:
                    matrix += ','

            matrix += ']'
            if not i == component_amount-1:
                matrix += ',\n'

        matrix += "]"

        start_conc = [0 for _ in range(component_amount)]
        start_conc[0] = 1

        print(matrix)
        return cls(matrix, start_conc=start_conc, solve_symbol=solve_symbol)


if __name__ == '__main__':

    f = KineticModel.evolution_associated_model(7, True)
    # Kinetic Matrix for Enzyme Kinetics à
    # [S] + [E] <-> [ES] -> [P] + [E]
    # matrix = (
    #     "[[-k0 * f1(t),           0,     +k1, 0],"  # [S]
    #     " [          0, -k0 * f0(t), +k1 +k2, 0],"  # [E]
    #     " [ k0 * f1(t),           0, -k1 -k2, 0],"  # [ES]
    #     " [ 0,                    0,     +k2, 0]]"  # [P]
    # )
