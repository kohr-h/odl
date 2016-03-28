.. _productspaces:

##############
Product Spaces
##############


Definition and basic properties
-------------------------------

A product space is a conceptually simple mathematical object that combines elements from different
vector spaces into a single element. However, the freedom to endorse this new space with
additional structure regarding the roles of the individual spaces in the definition of 
inner products, norms and distance functions makes product spaces a very interesting and
highly nontrivial object of study (especially if the number of member spaces is infinite).

In ODL, an arbitrary (however finite) number of arbitrary linear spaces can be combined to form
a product space. If :math:`\mathcal{X}_1, \dots, \mathcal{X}_n` are vector spaces, then the
corresponding product space is defined as

.. math::
    \prod_{i=1}^n \mathcal{X}_i := \mathcal{X}_1 \times \dots \mathcal{X}_n
    := \{ (x_1, \dots, x_n) | x_i \in \mathcal{X}_i,\ i=1, \dots, n \}.

Arithmetic operations on a product space are defined entry-wise, i.e. if 
:math:`x = (x_1, \dots, x_n)` and :math:`y = (y_1, \dots, y_n)` are elements of :math:`\mathcal{X}`,
it is

.. math::
    x + y = (x_1 + y_1, \dots, x_n + y_n).


Inner products, norms and distances
-----------------------------------
There is a canonical way to build inner products, norms and distance functions etc. from their
counterparts on the component spaces. If all spaces :math:`\mathcal{X}_i` are metric spaces equipped
with distance functions :math:`d_i`, one can first construct the :math:`n`-dimensional vector

.. math::
    D = \big( d_1(x_1, y_1), \dots, d_n(x_n, y_n) \big) \in \mathbb{R}^n

from :math:`x, y \in \mathcal{X}`. Now, any norm on :math:`\mathbb{R}^n` applied to :math:`D` defines
a valid distance function on :math:`\mathcal{X}`. Hence, given a norm :math:`\lVert \cdot \rVert`
on :math:`\mathbb{R}^n`, we can define

.. math::
    d(x, y) := \Big\lVert \big( d_i(x_i, y_i) \big)_{i=1}^n \Big\rVert.

Similarly, if the component spaces are equipped with norms :math:`\lVert \cdot \rVert_i`, a norm on
:math:`\mathcal{X}` can be defined as

.. math::
    \lVert x \rVert := \Big\lVert \big( \lVert x_i \rVert_i \big)_{i=1}^n \Big\rVert,

and the canonical distance function is :math:`d(x, y) = \lVert x - y \rVert` in this case. Finally,
for Hilbert spaces :math:`\mathcal{X}_i` with inner products :math:`\langle \cdot,\cdot\rangle_i`,
we get the natural inner product

.. math::
    \langle x, y \rangle := \sum_{i=1}^n \langle x_i, y_i \rangle_i

on the product space :math:`\mathcal{X}`.


Weighted spaces
---------------
TODO: link to Cartesian spaces

Due to the close link to Cartesian spaces, weightings on product spaces can be defined in a similar
way. Let :math:`\mathcal{X}` be as above, with component spaces over the field :math:`\mathbb{F}`.
Let further :math:`A \in \mathbb{F}^{n \times n}` and :math:`B \in \mathbb{R}^{n \times n}` be
Hermitian (symmetric) and positive definite matrices.

As for weighted Cartesian spaces, the obvious way to define the weighted norm of the vector of norms
:math:`N(x) = (\lVert x_i \rVert_i)_{i=1}^n` is

.. math:: \lVert N(x) \rVert_B := \lVert B N(x) \rVert_{\mathbb{R}^n},

and this can then be taken as the definition of the :math:`B`-weighted norm on :math:`\mathcal{X}`:

.. math:: \lVert x \rVert_{\mathcal{X}, B} := \lVert B N(x) \rVert_{\mathbb{R}^n}.

On the other hand, for :math:`p`-norms with :math:`p < \infty`, we rather set

.. math:: \lVert N(x) \rVert_{p, B} := \lVert B^{1/p} N(x) \rVert_p

and get the weighted norm

.. math:: \lVert x \rVert_{\mathcal{X}, p, B} := \lVert B^{1/p} N(x) \rVert_p.

For the inner product, we observe that

.. math::
    \langle x, y \rangle = \sum_{i=1}^n \langle x_i, y_i \rangle_i 
    = \langle S(x, y), \mathbf{1} \rangle_{\mathbb{F}^n}

with the vector :math:`S(x, y) := (\langle x_i, y_i \rangle_i)_{i=1}^n \in \mathbb{F}^n`
and :math:`\mathbf{1} = (1, \dots, 1)`. Hence, using the same argument as for Cartesian spaces, we
can acquire the weighted inner product

.. math::
    \langle x, y \rangle_A := \langle A S(x, y), \mathbf{1} \rangle_{\mathbb{F}^n}
    = \langle S(x, y), A \mathbf{1} \rangle_{\mathbb{F}^n}
    = \langle S(x, y), a \rangle_{\mathbb{F}^n},

where :math:`a \in \mathbb{F}^n` is the row-sum vector of the matrix :math:`A`.

Operators on product spaces
---------------------------
TODO: link to matrix adjoint

In further analogy to Cartesian spaces, weighting plays an important role for operators defined
between product spaces. As such, an operator :math:`\mathcal{M} : \mathcal{X} \to \mathcal{Y}` 
defined between product spaces :math:`\mathcal{X}` and :math:`\mathcal{Y}` is not conceptually
different from other operators. However, if :math:`\mathcal{M}` is a linear operator, and
:math:`\mathcal{X}` and :math:`\mathcal{Y}` have :math:`n` and :math:`m` components, respectively,
:math:`\mathcal{M}` can be uniquely written as an "operator matrix"

.. math::
    :nowrap:

    \begin{equation*}
      \mathcal{M} =
      \begin{pmatrix}
        \mathcal{M}_{11} & \hdots & \mathcal{M}_{1n} \\
        \vdots           & \ddots & \vdots           \\
        \mathcal{M}_{m1} & \hdots & \mathcal{M}_{mn}
      \end{pmatrix}
    \end{equation*}

For this case, the same rules regarding adjoints apply as in the case of the matrix operator, with
the difference that one takes the adjoint instead of the complex conjugate of each entry. If 
:math:`\mathbb{F}^{n \times n} \ni A = A^* \succeq 0` and
:math:`\mathbb{F}^{m \times m} \ni B = B^* \succeq 0` are the weighting matrices of the inner
products in :math:`\mathcal{X}` and :math:`\mathcal{Y}`, respectively, one gets with
:math:`a = A \mathbf{1}` and :math:`b = B \mathbf{1}`:

.. math::
    \langle \mathcal{M}(x), y \rangle_B 
    &= \big\langle S\big( \mathcal{M}(x), y \big), b \big\rangle_{\mathbb{F}^n}
    = \sum_{i=1}^m \big\langle \mathcal{M}(x)_i, y_i \big\rangle_{\mathcal{Y}_i} \, \overline{b_i} 
    = \sum_{i=1}^m \sum_{j=1}^n \big\langle \mathcal{M}_{ij}(x_j), y_i
    \big\rangle_{\mathcal{Y}_i} \, \overline{b_i} \\
    &= \sum_{i=1}^m \sum_{j=1}^n \big\langle x_j, \mathcal{M}_{ij}^*(y_i) 
    \big\rangle_{\mathcal{X}_j}\, \overline{b_i}
    = \sum_{i=1}^m \sum_{j=1}^n \big\langle x_j, b_i \mathcal{M}_{ij}^*(y_i) 
    \big\rangle_{\mathcal{X}_j} \\
    &= \sum_{j=1}^n \Big\langle x_j, \sum_{i=1}^m b_i \mathcal{M}_{ij}^*(y_i) 
    \Big\rangle_{\mathcal{X}_j} 
    = \sum_{j=1}^n (\overline{a_j})^{-1} \Big\langle x_j, \sum_{i=1}^m b_i \mathcal{M}_{ij}^*(y_i) 
    \Big\rangle_{\mathcal{X}_j} \, \overline{a_j} \\
    &= \sum_{j=1}^n \Big\langle x_j, a_j^{-1} \sum_{i=1}^m \mathcal{M}_{ij}^*(b_i y_i) 
    \Big\rangle_{\mathcal{X}_j} \, \overline{a_j}
    = \langle x, \mathcal{M}^*(y) \rangle_A,

where we can identify the adjoint operator

.. math::
    :nowrap:

    \begin{equation*}
      \mathcal{M}^* = \mathrm{diag}(a_1^{-1}, \dots, a_n^{-1}) \cdot 
      \begin{pmatrix}
        \mathcal{M}_{11}^* & \hdots & \mathcal{M}_{m1}^* \\
        \vdots             & \ddots & \vdots             \\
        \mathcal{M}_{1n}^* & \hdots & \mathcal{M}_{mn}^*
      \end{pmatrix}
      \cdot \mathrm{diag}(b_1, \dots, b_m).
    \end{equation*}
    

Useful Wikipedia articles
-------------------------

- `Product Topology`_
- `Vector space norms`_


.. _Vector space norms: https://en.wikipedia.org/wiki/Norm_%28mathematics%29
.. _Product Topology: https://en.wikipedia.org/wiki/Product_topology
