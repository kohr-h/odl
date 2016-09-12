# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Test for the Functional class."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np
import pytest

# Internal
import odl
from odl.util.testutils import all_almost_equal, almost_equal, noise_element

# Places for the accepted error when comparing results
PLACES = 8


# TODO: maybe add tests for if translations etc. belongs to the wrong space.
# These tests doesn't work as intended now, since casting is possible between
# spaces with the same number of discrete points.

space_params = ['r10', 'uniform_discr']
space_ids = [' space = {}'.format(p.ljust(10)) for p in space_params]


@pytest.fixture(scope="module", ids=space_ids, params=space_params)
def space(request, fn_impl):
    name = request.param.strip()

    if name == 'r10':
        return odl.rn(10, impl=fn_impl)
    elif name == 'uniform_discr':
        # Discretization parameters
        return odl.uniform_discr(0, 1, 7, impl=fn_impl)


func_params = ['l1 ', 'l2', 'l2^2', 'constant']
func_ids = [' f = {}'.format(p.ljust(10)) for p in func_params]


@pytest.fixture(scope="module", ids=func_ids, params=func_params)
def functional(request, space):
    name = request.param.strip()

    if name == 'l1':
        func = odl.solvers.functional.L1Norm(space)
    elif name == 'l2':
        func = odl.solvers.functional.L2Norm(space)
    elif name == 'l2^2':
        func = odl.solvers.functional.L2NormSquared(space)
    elif name == 'constant':
        func = odl.solvers.functional.ConstantFunctional(space, 2)

    return func


def test_derivative(functional):
    """Test for the derivative of a functional.

    The test checks that the directional derivative in a point is the same as
    the inner product of the gradient and the direction, if the gradient is
    defined.
    """

    x = noise_element(functional.domain)
    y = noise_element(functional.domain)
    epsK = 1e-8

    # Numerical test of gradient
    assert all_almost_equal((functional(x + epsK * y) - functional(x)) / epsK,
                            y.inner(functional.gradient(x)),
                            places=PLACES / 2)

    # Check that derivative and gradient is consistent
    assert all_almost_equal(functional.derivative(x)(y),
                            y.inner(functional.gradient(x)),
                            places=PLACES)


def test_left_scalar_multiplication():
    """Test for right and left multiplication of a functional with a scalar."""

    # Discretization parameters
    n = 3

    # Discretized spaces
    space = odl.uniform_discr([0, 0], [1, 1], [n, n])
    x = noise_element(space)

    scal = np.random.standard_normal()
    F = odl.solvers.functional.L2Norm(space)

    # Evaluation of left scalar multiplication
    assert all_almost_equal((scal * F)(x), scal * (F(x)),
                            places=PLACES)

    # Test gradient of left scalar multiplication
    assert all_almost_equal((scal * F).gradient(x), scal * (F.gradient(x)),
                            places=PLACES)

    # Test derivative of left scalar multiplication
    p = noise_element(space)
    assert all_almost_equal(((scal * F).derivative(x))(p),
                            scal * ((F.derivative(x))(p)),
                            places=PLACES)

    # Test conjugate functional. This requiers positive scaling to work
    scal = np.random.rand()
    neg_scal = -np.random.rand()

    with pytest.raises(ValueError):
        (neg_scal * F).convex_conj

    assert all_almost_equal((scal * F).convex_conj(x),
                            scal * (F.convex_conj(x / scal)),
                            places=PLACES)

    # TODO: Add more test for convex conjugate? Old ConvexConjugateArgScaling?

    # Test proximal operator. This requiers scaling to be positive.
    sigma = 1.0
    with pytest.raises(ValueError):
        (neg_scal * F).proximal(sigma)

    sigma = np.random.rand()
    assert all_almost_equal(((scal * F).proximal(sigma))(x),
                            (F.proximal(sigma * scal))(x),
                            places=PLACES)

    # Test left multiplication with zero
    zero_times_f = 0 * F
    x = noise_element(space)
    assert all_almost_equal(zero_times_f(x), space.zero(), places=PLACES)

    # Proximal of the zero functional is the identity operator
    sigma = np.random.rand()
    zero_prox = zero_times_f.proximal(sigma)
    x_verify = x.copy()
    assert all_almost_equal(zero_prox(x), x_verify, places=PLACES)


def test_right_scalar_multiplication():
    """Test for right and left multiplication of a functional with a scalar."""

    # Discretization parameters
    n = 3

    # Discretized spaces
    space = odl.uniform_discr([0, 0], [1, 1], [n, n])
    x = noise_element(space)

    scal = np.random.standard_normal()
    F = odl.solvers.functional.L2NormSquared(space)

    # Evaluation of right scalar multiplication
    assert all_almost_equal((F * scal)(x), (F)(scal * x),
                            places=PLACES)

    # Test gradient of right scalar multiplication
    assert all_almost_equal((F * scal).gradient(x),
                            scal * (F.gradient(scal * x)),
                            places=PLACES)

    # Test derivative of right scalar multiplication
    p = noise_element(space)
    assert all_almost_equal(((F * scal).derivative(x))(p),
                            scal * (F.derivative(scal * x))(p),
                            places=PLACES)

    # Test conjugate functional. This requiers positive scaling to work
    scal = np.random.rand()
    neg_scal = -np.random.rand()

    with pytest.raises(ValueError):
        (F * neg_scal).convex_conj

    assert all_almost_equal((F * scal).convex_conj(x),
                            (F.convex_conj(x / scal)),
                            places=PLACES)

    # TODO: Add more test for convex conjugate? Old ConvexConjugateFuncScaling?

    # Test proximal operator. This requiers scaling to be positive.
    sigma = np.random.rand()
    assert all_almost_equal(((F * scal).proximal(sigma))(x),
                            ((1.0 / scal) *
                                (F.proximal(sigma * scal**2)))(x * scal),
                            places=PLACES)

    # Test that for linear functionals, left multiplication is used.
    func = odl.solvers.ZeroFunctional(space)
    assert isinstance(scal * func, odl.solvers.FunctionalLeftScalarMult)


def test_functional_composition():
    """Test composition from the right with an operator."""

    space = odl.uniform_discr(0, 1, 10)
    func = odl.solvers.L2NormSquared(space)

    # Test composition with operator from the right
    scalar = np.random.rand()
    wrong_space = odl.uniform_discr(1, 2, 10)
    op_wrong = odl.operator.ScalingOperator(wrong_space, scalar)

    with pytest.raises(TypeError):
        func * op_wrong

    op = odl.operator.ScalingOperator(space, scalar)
    assert isinstance(func * op, odl.solvers.Functional)

    x = noise_element(space)
    assert almost_equal((func * op)(x), func(op(x)), places=PLACES)

    # Test gradient and derivative with composition from the right
    assert all_almost_equal(((func * op).gradient)(x),
                            (op.adjoint * func.gradient * op)(x),
                            places=PLACES)

    p = noise_element(space)
    assert all_almost_equal((func * op).derivative(x)(p),
                            (op.adjoint * func.gradient * op)(x).inner(p),
                            places=PLACES)


def test_functional_sum():
    """Test for the sum of two functionals."""
    space = odl.uniform_discr(0, 1, 10)

    func1 = odl.solvers.L2NormSquared(space)
    func2 = odl.solvers.L2Norm(space)

    # Test for sum where one is not a functional
    op = odl.operator.IdentityOperator(space)
    with pytest.raises(TypeError):
        func1 + op

    # Test for different domain of the functionals
    wrong_space = odl.uniform_discr(1, 2, 10)
    func_wrong_domain = odl.solvers.L2Norm(wrong_space)
    with pytest.raises(TypeError):
        func1 + func_wrong_domain

    x = noise_element(space)
    p = noise_element(space)

    # Test evaluation of the functionals
    assert almost_equal((func1 + func2)(x),
                        func1(x) + func2(x),
                        places=PLACES)

    # Test for the gradient and derivative
    assert all_almost_equal((func1 + func2).gradient(x),
                            func1.gradient(x) + func2.gradient(x),
                            places=PLACES)

    assert almost_equal((func1 + func2).derivative(x)(p),
                        (func1.gradient(x).inner(p) +
                            func2.gradient(x).inner(p)),
                        places=PLACES)

    # Test that prox and convex conjugate is not known
    with pytest.raises(NotImplementedError):
        (func1 + func2).proximal
    with pytest.raises(NotImplementedError):
        (func1 + func2).convex_conj


def test_functional_plus_scalar():
    """Test for sum of functioanl and scalar."""
    space = odl.uniform_discr(0, 1, 10)

    func = odl.solvers.L2NormSquared(space)
    scalar = np.random.randn()

    # Test for scalar not in the field (field of unifor_discr is RealNumbers)
    complex_scalar = 1j
    with pytest.raises(TypeError):
        func + complex_scalar

    x = noise_element(space)
    p = noise_element(space)

    # Test for evaluation
    assert almost_equal((func + scalar)(x), func(x) + scalar, places=PLACES)

    # Test for derivative and gradient
    assert all_almost_equal((func + scalar).gradient(x),
                            func.gradient(x), places=PLACES)

    assert almost_equal((func + scalar).derivative(x)(p),
                        func.gradient(x).inner(p),
                        places=PLACES)

    # Test proximal operator
    sigma = np.random.rand()
    assert all_almost_equal((func + scalar).proximal(sigma)(x),
                            func.proximal(sigma)(x), places=PLACES)

    # Test convex conjugate functional
    assert almost_equal((func + scalar).convex_conj(x),
                        func.convex_conj(x) - scalar, places=PLACES)

    assert all_almost_equal((func + scalar).convex_conj.gradient(x),
                            func.convex_conj.gradient(x),
                            places=PLACES)


def test_translation_of_functional():
    """Test for the translation of a functional: (f(. - y))^*"""
    space = odl.uniform_discr(0, 1, 10)

    # The translation; an element in the domain
    translation = noise_element(space)

    # Creating the functional ||x||_2^2
    test_functional = odl.solvers.L2NormSquared(space)

    # Create translated functional
    translated_functional = test_functional.translated(translation)

    # Create an element in the space, in which to evaluate
    x = noise_element(space)

    # Test for evaluation of the functional
    expected_result = test_functional(x - translation)
    assert all_almost_equal(translated_functional(x), expected_result,
                            places=PLACES)

    # Test for the gradient
    expected_result = test_functional.gradient(x - translation)
    translated_gradient = translated_functional.gradient
    assert all_almost_equal(translated_gradient(x), expected_result,
                            places=PLACES)

    # Test for proximal
    sigma = np.random.rand()
    # The helper function below is tested explicitly in proximal_utils_test
    expected_result = odl.solvers.proximal_translation(
        test_functional.proximal, translation)(sigma)(x)
    assert all_almost_equal(translated_functional.proximal(sigma)(x),
                            expected_result, places=PLACES)

    # Test for conjugate functional
    # The helper function below is tested explicitly further down in this file
    expected_result = odl.solvers.FunctionalLinearPerturb(
        test_functional.convex_conj, translation)(x)
    assert all_almost_equal(translated_functional.convex_conj(x),
                            expected_result, places=PLACES)

    # Test for derivative in direction p
    p = noise_element(space)

    # Explicit computation in point x, in direction p: <x/2 + translation, p>
    expected_result = p.inner(test_functional.gradient(x - translation))
    assert all_almost_equal(translated_functional.derivative(x)(p),
                            expected_result,
                            places=PLACES)

    # Test for optimized implementation, when translating a translated
    # functional
    second_translation = noise_element(space)
    double_translated_functional = translated_functional.translated(
        second_translation)

    # Evaluation
    assert almost_equal(double_translated_functional(x),
                        test_functional(x - translation - second_translation),
                        places=PLACES)


def test_multiplication_with_vector():
    """Test for multiplying a functional with a vector, both left and right."""

    space = odl.uniform_discr(0, 1, 10)

    x = noise_element(space)
    y = noise_element(space)
    func = odl.solvers.L2NormSquared(space)

    wrong_space = odl.uniform_discr(1, 2, 10)
    y_other_space = noise_element(wrong_space)

    # Multiplication from the right. Make sure it is a
    # FunctionalRightVectorMult
    func_times_y = func * y
    assert isinstance(func_times_y, odl.solvers.FunctionalRightVectorMult)

    expected_result = func(y * x)
    assert almost_equal(func_times_y(x), expected_result, places=PLACES)

    # Test for the gradient.
    # Explicit calculations: 2*y*y*x
    expected_result = 2.0 * y * y * x
    assert all_almost_equal(func_times_y.gradient(x), expected_result,
                            places=PLACES)

    # Test for convex_conj
    cc_func_times_y = func_times_y.convex_conj
    # Explicit calculations: 1/4 * ||x/y||_2^2
    expected_result = 1.0 / 4.0 * (x / y).norm()**2
    assert almost_equal(cc_func_times_y(x), expected_result, places=PLACES)

    # Make sure that right muliplication is not allowed with vector from
    # another space
    with pytest.raises(TypeError):
        func * y_other_space

    # Multiplication from the left. Make sure it is a FunctionalLeftVectorMult
    y_times_func = y * func
    assert isinstance(y_times_func, odl.FunctionalLeftVectorMult)

    expected_result = y * func(x)
    assert all_almost_equal(y_times_func(x), expected_result, places=PLACES)

    # Now, multiplication with vector from another space is ok (since it is the
    # same as scaling that vector with the scalar returned by the functional).
    y_other_times_func = y_other_space * func
    assert isinstance(y_other_times_func, odl.FunctionalLeftVectorMult)

    expected_result = y_other_space * func(x)
    assert all_almost_equal(y_other_times_func(x), expected_result,
                            places=PLACES)


def test_functional_linear_perturb():
    """Test for the functional f(.) + <y, .>."""
    space = odl.uniform_discr(0, 1, 10)

    # The translation; an element in the domain
    linear_term = noise_element(space)

    # Creating the functional ||x||_2^2 and add the linear perturbation
    orig_func = odl.solvers.L2NormSquared(space)
    functional = odl.solvers.FunctionalLinearPerturb(orig_func, linear_term)

    # Create an element in the space, in which to evaluate
    x = noise_element(space)

    # Test for evaluation of the functional
    assert all_almost_equal(functional(x), x.norm()**2 + x.inner(linear_term),
                            places=PLACES)

    # Test for the gradient
    assert all_almost_equal(functional.gradient(x), 2.0 * x + linear_term,
                            places=PLACES)

    # Test for derivative in direction p
    p = noise_element(space)
    assert all_almost_equal(functional.derivative(x)(p),
                            p.inner(2 * x + linear_term),
                            places=PLACES)

    # Test for the proximal operator
    sigma = np.random.rand()
    # Explicit computation gives (x - sigma * translation)/(2 * sigma + 1)
    expected_result = (x - sigma * linear_term) / (2.0 * sigma + 1.0)
    assert all_almost_equal(functional.proximal(sigma)(x), expected_result,
                            places=PLACES)

    # Test convex conjugate functional
    assert almost_equal(functional.convex_conj(x),
                        orig_func.convex_conj.translated(linear_term)(x),
                        places=PLACES)

    # Test proximal of the convex conjugate
    assert all_almost_equal(
        functional.convex_conj.proximal(sigma)(x),
        orig_func.convex_conj.translated(linear_term).proximal(sigma)(x),
        places=PLACES)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
