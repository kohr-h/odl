# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for the default functionals."""

from __future__ import division

import numpy as np
import pytest
import scipy.special

import odl
from odl.solvers.functional.default_functionals import (
    KullbackLeiblerConvexConj, KullbackLeiblerCrossEntropyConvexConj)
from odl.util.testutils import all_almost_equal, noise_element, simple_fixture

# --- pytest fixtures --- #


scalar = simple_fixture(
    'scalar', [0.01, 2.7, np.array(5.0), 10, -2, -0.2, -np.array(7.1), 0]
)
sigma = simple_fixture('sigma', [0.001, 2.7, 10])
exponent = simple_fixture('exponent', [1, 2, 1.5, 2.5])


space_params = ['r10', 'Lp', 'Lp ** 2']
space_ids = [' space={} '.format(p) for p in space_params]


@pytest.fixture(scope="module", ids=space_ids, params=space_params)
def space(request, odl_tspace_impl):
    name = request.param.strip()
    impl = odl_tspace_impl

    if name == 'r10':
        return odl.rn(10, impl=impl)
    elif name == 'Lp':
        return odl.uniform_discr(0, 1, 7, impl=impl)
    elif name == 'Lp ** 2':
        space = odl.uniform_discr(0, 1, 7, impl=impl)
        return odl.ProductSpace(space, 2)

# --- functional tests --- #


def test_L1_norm(space, sigma):
    """Test the L1-norm."""
    func = odl.solvers.L1Norm(space)
    x = noise_element(space)
    F = space.ufuncs
    R = space.reduce
    sigma = float(sigma)

    assert func(x) == pytest.approx(space.inner(F.abs(x), space.one()))
    assert all_almost_equal(func.gradient(x), F.sign(x))
    assert all_almost_equal(
        func.proximal(sigma)(x), F.sign(x) * F.maximum(F.abs(x) - sigma, 0)
    )

    func_cc = func.convex_conj
    inf_norm_x = R.max(F.abs(x))
    for c in [2.0, 1.1, 1 + 1e-5]:
        norm_gt_1 = (c / inf_norm_x) * x
        assert func_cc(norm_gt_1) == np.inf

    for c in [0.0, 0.9, 1 - 1e-5]:
        norm_lt_1 = (c / inf_norm_x) * x
        assert func_cc(norm_lt_1) == 0

    assert all_almost_equal(
        func_cc.proximal(sigma)(x), x / F.maximum(1, F.abs(x))
    )

    func_cc_cc = func_cc.convex_conj
    assert isinstance(func_cc_cc, odl.solvers.L1Norm)


def test_indicator_lp_unit_ball(space, sigma, exponent):
    """Test for indicator function on unit ball."""
    func = odl.solvers.IndicatorLpUnitBall(space, exponent)
    x = noise_element(space)
    F = space.ufuncs

    p_norm_x = np.power(
        space.inner(F.power(F.abs(x), exponent), space.one()), 1 / exponent
    )
    for c in [2.0, 1.1, 1 + 1e-5]:
        norm_gt_1 = (c / p_norm_x) * x
        assert func(norm_gt_1) == np.inf

    for c in [0.0, 0.9, 1 - 1e-5]:
        norm_lt_1 = (c / p_norm_x) * x
        assert func(norm_lt_1) == 0


def test_L2_norm(space, sigma):
    """Test the L2-norm."""
    func = odl.solvers.L2Norm(space)
    x = noise_element(space)
    x_norm = space.norm(x)
    zero = space.zero()

    assert func(x) == pytest.approx(np.sqrt(space.inner(x ** 2, space.one())))
    if x_norm > 0:
        assert all_almost_equal(func.gradient(x), x / x_norm)
    assert all_almost_equal(func.gradient(zero), zero)

    # Prox: x * (1 - sigma/||x||) if ||x|| > sigma, else 0
    for c in [2.0, 1.1, 1 + 1e-5]:
        norm_gt_sig = (c * sigma / x_norm) * x
        expected_result = (
            norm_gt_sig * (1.0 - sigma / space.norm(norm_gt_sig))
        )
        assert all_almost_equal(
            func.proximal(sigma)(norm_gt_sig), expected_result
        )

    for c in [0.0, 0.9, 1 - 1e-5]:
        norm_lt_sig = (c * sigma / x_norm) * x
        assert all_almost_equal(func.proximal(sigma)(norm_lt_sig), zero)

    func_cc = func.convex_conj
    # Convex conj: 0 if ||x|| < 1, else infty
    for c in [2.0, 1.1, 1 + 1e-5]:
        norm_gt_1 = (c / x_norm) * x
        assert func_cc(norm_gt_1) == np.inf

    for c in [0.0, 0.9, 1 - 1e-5]:
        norm_lt_1 = (c / x_norm) * x
        assert func_cc(norm_lt_1) == 0

    # Convex conj prox: x if ||x||_2 < 1, else x/||x||
    for c in [2.0, 1.1, 1 + 1e-5]:
        norm_gt_1 = (c / x_norm) * x
        assert all_almost_equal(
            func_cc.proximal(sigma)(norm_gt_1), x / x_norm
        )

    for c in [0.0, 0.9, 1 - 1e-5]:
        print(c)
        norm_lt_1 = (c / x_norm) * x
        assert all_almost_equal(
            func_cc.proximal(sigma)(norm_lt_1), norm_lt_1
        )

    func_cc_cc = func_cc.convex_conj
    assert func_cc_cc(x) == pytest.approx(func(x))


def test_L2_norm_squared(space, sigma):
    """Test the squared L2-norm."""
    func = odl.solvers.L2NormSquared(space)
    x = noise_element(space)
    x_norm = space.norm(x)

    assert func(x) == pytest.approx(x_norm ** 2)
    assert all_almost_equal(func.gradient(x), 2 * x)
    assert all_almost_equal(func.proximal(sigma)(x), x / (1 + 2 * sigma))

    func_cc = func.convex_conj
    assert func_cc(x) == pytest.approx(x_norm ** 2 / 4)
    assert all_almost_equal(func_cc.gradient(x), x / 2.0)
    assert all_almost_equal(func_cc.proximal(sigma)(x), x / (1 + sigma / 2.0))

    func_cc_cc = func_cc.convex_conj
    assert func_cc_cc(x) == pytest.approx(func(x))
    assert all_almost_equal(func_cc_cc.gradient(x), func.gradient(x))


def test_constant_functional(space, scalar):
    """Test the constant functional."""
    constant = float(scalar)
    func = odl.solvers.ConstantFunctional(space, constant=scalar)
    x = noise_element(space)
    sigma = 1.5

    assert func(x) == constant
    assert isinstance(func.gradient, odl.ZeroOperator)
    assert isinstance(func.proximal(sigma), odl.IdentityOperator)

    func_cc = func.convex_conj
    # Convex conj: -constant if x = 0, else infty
    assert func_cc(x) == np.inf
    assert func_cc(space.zero()) == -constant
    assert isinstance(func_cc.proximal(sigma), odl.ZeroOperator)

    func_cc_cc = func_cc.convex_conj
    assert isinstance(func_cc_cc, odl.solvers.ConstantFunctional)
    assert func_cc_cc.constant == constant


def test_zero_functional(space):
    """Test the zero functional."""
    zero_func = odl.solvers.ZeroFunctional(space)
    assert zero_func(space.one()) == 0


def test_kullback_leibler(space):
    """Test the kullback leibler functional and its convex conjugate."""
    F = space.ufuncs
    R = space.reduce
    prior = noise_element(space)
    prior = space.element(F.abs(prior) + 0.1)  # must be positive
    func = odl.solvers.KullbackLeibler(space, prior)
    x = noise_element(space)
    x = func.domain.element(np.abs(x) + 0.1)  # must be positive
    sigma = 1.2
    one = space.one()

    assert func(x) == pytest.approx(
        space.inner((x - prior + prior * F.log(prior / x)), one)
    )

    # If one or more components are negative, the result should be infinity
    x_neg = noise_element(space)
    x_neg = x_neg - R.max(x_neg)
    assert func(x_neg) == np.inf

    assert all_almost_equal(func.gradient(x), 1 - prior / x)
    # TODO(kohr-h): this just tests the implementation, not the result
    expected_result = odl.solvers.proximal_convex_conj(
        odl.solvers.proximal_convex_conj_kl(space, g=prior))(sigma)(x)
    assert all_almost_equal(func.proximal(sigma)(x), expected_result)

    func_cc = func.convex_conj

    # Convex conjugate is the integral of -prior * log(1 - x) if x < 1
    # everywhere, otherwise infinity
    x_lt_1 = x - R.max(x) + 0.99
    assert func_cc(x_lt_1) == pytest.approx(
        -space.inner(prior * F.log(1 - x_lt_1), one)
    )
    x_ge_1 = x - R.max(x) + 1.01
    assert func_cc(x_ge_1) == np.inf

    assert all_almost_equal(func_cc.gradient(x_lt_1), prior / (1 - x_lt_1))

    expected_result = (
        1 + x_lt_1 - F.sqrt((x_lt_1 - 1) ** 2 + 4 * sigma * prior)
    ) / 2
    assert all_almost_equal(
        func_cc.proximal(sigma)(x_lt_1), expected_result
    )

    func_cc_cc = func_cc.convex_conj
    assert func_cc_cc(x) == pytest.approx(func(x))


def test_kullback_leibler_cross_entorpy(space):
    """Test the kullback leibler cross entropy and its convex conjugate."""
    # The prior needs to be positive
    prior = noise_element(space)
    prior = space.element(np.abs(prior))

    func = odl.solvers.KullbackLeiblerCrossEntropy(space, prior)

    # The fucntional is only defined for positive elements
    x = noise_element(space)
    x = func.domain.element(np.abs(x))
    one_elem = space.one()

    # Evaluation of the functional
    expected_result = space.inner(
        (prior - x + x * np.log(x / prior)), one_elem
    )
    assert func(x) == pytest.approx(expected_result)

    # Check property for prior
    assert all_almost_equal(func.prior, prior)

    # For elements with (a) negative components it should return inf
    x_neg = noise_element(space)
    x_neg = x_neg - np.max(x_neg)
    assert func(x_neg) == np.inf

    # The gradient
    expected_result = np.log(x / prior)
    assert all_almost_equal(func.gradient(x), expected_result)

    # The proximal operator
    sigma = np.random.rand()
    prox = odl.solvers.proximal_convex_conj(
        odl.solvers.proximal_convex_conj_kl_cross_entropy(space, g=prior))
    expected_result = prox(sigma)(x)
    assert all_almost_equal(func.proximal(sigma)(x), expected_result)

    # The convex conjugate functional
    cc_func = func.convex_conj

    assert isinstance(cc_func, KullbackLeiblerCrossEntropyConvexConj)

    # The convex conjugate functional is defined for all values of x.
    x = noise_element(space)

    # Evaluation of convex conjugate
    expected_result = space.inner(prior * (np.exp(x) - 1), one_elem)
    assert cc_func(x) == pytest.approx(expected_result)

    # The gradient of the convex conjugate
    expected_result = prior * np.exp(x)
    assert all_almost_equal(cc_func.gradient(x), expected_result)

    # The proximal of the convex conjugate
    expected_result = (
        x - scipy.special.lambertw(sigma * prior * np.exp(x)).real
    )
    assert all_almost_equal(cc_func.proximal(sigma)(x), expected_result)

    # The biconjugate, which is the functional itself since it is proper,
    # convex and lower-semicontinuous
    cc_cc_func = cc_func.convex_conj

    # Check that they evaluate the same
    assert cc_cc_func(x) == pytest.approx(func(x))


def test_quadratic_form(space):
    """Test the quadratic form functional."""
    operator = odl.IdentityOperator(space)
    vector = space.one()
    constant = 0.363
    func = odl.solvers.QuadraticForm(operator, vector, constant)

    x = noise_element(space)

    # Checking that values is stored correctly
    assert func.operator == operator
    assert func.vector == vector
    assert func.constant == constant

    # Evaluation of the functional
    expected_result = (
        space.inner(x, operator(x)) + space.inner(x, vector) + constant
    )
    assert func(x) == pytest.approx(expected_result)

    # The gradient
    expected_gradient = 2 * operator(x) + vector
    assert all_almost_equal(func.gradient(x), expected_gradient)

    # The convex conjugate
    assert isinstance(func.convex_conj, odl.solvers.QuadraticForm)

    # Test for linear functional
    func_no_operator = odl.solvers.QuadraticForm(vector=vector,
                                                 constant=constant)
    expected_result = space.inner(x, vector) + constant
    assert func_no_operator(x) == pytest.approx(expected_result)

    expected_gradient = vector
    assert all_almost_equal(func_no_operator.gradient(x), expected_gradient)

    # The convex conjugate is a translation of the IndicatorZero
    func_no_operator_cc = func_no_operator.convex_conj
    assert isinstance(func_no_operator_cc,
                      odl.solvers.FunctionalTranslation)
    assert isinstance(func_no_operator_cc.functional,
                      odl.solvers.IndicatorZero)
    assert func_no_operator_cc(vector) == -constant
    assert np.isinf(func_no_operator_cc(vector + 2.463))

    # Test with no offset
    func_no_offset = odl.solvers.QuadraticForm(operator, constant=constant)
    expected_result = space.inner(x, operator(x)) + constant
    assert func_no_offset(x) == pytest.approx(expected_result)


def test_separable_sum(space):
    """Test for the separable sum."""
    l1 = odl.solvers.L1Norm(space)
    l2 = odl.solvers.L2Norm(space)

    x = noise_element(space)
    y = noise_element(space)

    # Initialization and calling
    func = odl.solvers.SeparableSum(l1, l2)
    assert func([x, y]) == pytest.approx(l1(x) + l2(y))

    power_func = odl.solvers.SeparableSum(l1, 5)
    assert power_func([x, x, x, x, x]) == pytest.approx(5 * l1(x))

    # Gradient
    grad = func.gradient([x, y])
    assert grad[0] == l1.gradient(x)
    assert grad[1] == l2.gradient(y)

    # Proximal
    sigma = 1.0
    prox = func.proximal(sigma)([x, y])
    assert prox[0] == l1.proximal(sigma)(x)
    assert prox[1] == l2.proximal(sigma)(y)

    # Convex conjugate
    assert func.convex_conj([x, y]) == l1.convex_conj(x) + l2.convex_conj(y)


def test_moreau_envelope_l1():
    """Test for the Moreau envelope with L1 norm."""
    space = odl.rn(3)

    l1 = odl.solvers.L1Norm(space)

    # Test l1 norm, gives "Huber norm"
    smoothed_l1 = odl.solvers.MoreauEnvelope(l1)
    assert all_almost_equal(smoothed_l1.gradient([0, -0.2, 0.7]),
                            [0, -0.2, 0.7])
    assert all_almost_equal(smoothed_l1.gradient([-3, 2, 10]),
                            [-1, 1, 1])

    # Test with different sigma
    smoothed_l1 = odl.solvers.MoreauEnvelope(l1, sigma=0.5)
    assert all_almost_equal(smoothed_l1.gradient([0, 0.2, 0.7]),
                            [0, 0.4, 1.0])


def test_moreau_envelope_l2_sq(space, sigma):
    """Test for the Moreau envelope with l2 norm squared."""

    # Result is ||x||_2^2 / (1 + 2 sigma)
    # Gradient is x * 2 / (1 + 2 * sigma)
    l2_sq = odl.solvers.L2NormSquared(space)

    smoothed_l2_sq = odl.solvers.MoreauEnvelope(l2_sq, sigma=sigma)
    x = noise_element(space)
    assert all_almost_equal(smoothed_l2_sq.gradient(x),
                            x * 2 / (1 + 2 * sigma))


def test_weighted_separablesum(space):
    """Test for the weighted proximal of a SeparableSum functional."""

    l1 = odl.solvers.L1Norm(space)
    l2 = odl.solvers.L2Norm(space)
    func = odl.solvers.SeparableSum(l1, l2)

    x = func.domain.one()

    sigma = [0.5, 1.0]

    prox = func.proximal(sigma)(x)
    assert all_almost_equal(prox, [l1.proximal(sigma[0])(x[0]),
                                   l2.proximal(sigma[1])(x[1])])


def test_weighted_proximal_L2_norm_squared(space):
    """Test for the weighted proximal of the squared L2 norm"""

    # Define the functional on the space.
    func = odl.solvers.L2NormSquared(space)

    # Set the stepsize as a random element of the spaces
    # with elements between 1 and 10.
    sigma = odl.phantom.uniform_noise(space, 1, 10)

    # Start at the one vector.
    x = space.one()

    # Calculate the proximal point in-place and out-of-place
    p_ip = space.element()
    func.proximal(sigma)(x, out=p_ip)
    p_oop = func.proximal(sigma)(x)

    # Both should contain the same vector now.
    assert all_almost_equal(p_ip, p_oop)

    # Check if the subdifferential inequalities are satisfied.
    # p = prox_{sigma * f}(x) iff (x - p)/sigma = grad f(p)
    assert all_almost_equal(func.gradient(p_ip),
                            (x - p_ip) / sigma)


def test_weighted_proximal_L1_norm_far(space):
    """Test for the weighted proximal of the L1 norm away from zero"""

    # Define the functional on the space.
    func = odl.solvers.L1Norm(space)

    # Set the stepsize as a random element of the spaces
    # with elements between 1 and 10.
    sigma = odl.phantom.noise.uniform_noise(space, 1, 10)

    # Start far away from zero so that the L1 norm will be differentiable
    # at the result.
    x = 100 * space.one()

    # Calculate the proximal point in-place and out-of-place
    p_ip = space.element()
    func.proximal(sigma)(x, out=p_ip)
    p_oop = func.proximal(sigma)(x)

    # Both should contain the same vector now.
    assert all_almost_equal(p_ip, p_oop)

    # Check if the subdifferential inequalities are satisfied.
    # p = prox_{sigma * f}(x) iff (x - p)/sigma = grad f(p)
    assert all_almost_equal(func.gradient(p_ip), (x - p_ip) / sigma)


def test_weighted_proximal_L1_norm_close(space):
    """Test for the weighted proximal of the L1 norm near zero"""

    # Set the space.
    space = odl.rn(5)

    # Define the functional on the space.
    func = odl.solvers.L1Norm(space)

    # Set the stepsize.
    sigma = [0.1, 0.2, 0.5, 1.0, 2.0]

    # Set the starting point.
    x = 0.5 * space.one()

    # Calculate the proximal point in-place and out-of-place
    p_ip = space.element()
    func.proximal(sigma)(x, out=p_ip)
    p_oop = func.proximal(sigma)(x)

    # Both should contain the same vector now.
    assert all_almost_equal(p_ip, p_oop)

    # Check if this equals the expected result.
    expected_result = [0.4, 0.3, 0.0, 0.0, 0.0]
    assert all_almost_equal(expected_result, p_ip)


def test_bregman_functional_no_gradient(space):
    """Test Bregman distance for functional without gradient."""

    ind_func = odl.solvers.IndicatorNonnegativity(space)
    point = np.abs(noise_element(space))
    subgrad = noise_element(space)  # Any element in the domain is ok
    bregman_dist = odl.solvers.BregmanDistance(ind_func, point, subgrad)

    x = np.abs(noise_element(space))

    expected_result = space.inner(-subgrad, x - point)
    assert all_almost_equal(bregman_dist(x), expected_result)

    # However, since the functional is not differentialbe we cannot call the
    # gradient of the Bregman distance functional
    with pytest.raises(NotImplementedError):
        bregman_dist.gradient


def test_bregman_functional_l2_squared(space, sigma):
    """Test Bregman distance using l2 norm squared as underlying functional."""
    sigma = float(sigma)

    l2_sq = odl.solvers.L2NormSquared(space)
    point = noise_element(space)
    subgrad = l2_sq.gradient(point)
    bregman_dist = odl.solvers.BregmanDistance(l2_sq, point, subgrad)

    expected_func = odl.solvers.L2NormSquared(space).translated(point)

    x = noise_element(space)

    # Function evaluation
    assert all_almost_equal(bregman_dist(x), expected_func(x))

    # Gradient evaluation
    assert all_almost_equal(bregman_dist.gradient(x),
                            expected_func.gradient(x))

    # Convex conjugate
    cc_bregman_dist = bregman_dist.convex_conj
    cc_expected_func = expected_func.convex_conj
    assert all_almost_equal(cc_bregman_dist(x), cc_expected_func(x))

    # Proximal operator
    prox_bregman_dist = bregman_dist.proximal(sigma)
    prox_expected_func = expected_func.proximal(sigma)
    assert all_almost_equal(prox_bregman_dist(x), prox_expected_func(x))


if __name__ == '__main__':
    odl.util.test_file(__file__)
