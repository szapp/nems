"""
Nonlinearities and their derivatives

Each function returns the value and derivative of a nonlinearity. Given :math:`y = f(x)`, the function returns
:math:`y` and :math:`dy/dx`
"""
import numpy as np


def exp(x):
    """Exponential function"""

    # compute the exponential
    y = np.exp(x)

    # compute the first and second derivatives
    dydx = y
    dy2dx2 = y

    return y, dydx, dy2dx2


def softrect(x, a1=1, a2=1, a3=0):
    """ Soft rectifying function

    .. math::
        y = a1\log(1+e^(a2*(x+a3)))
    """

    # compute the soft rectifying nonlinearity
    x_exp = np.exp(a2*(x+a3))
    y = a1 * np.log1p(x_exp)

    # compute the derivative
    dydx = (a1 * a2 * x_exp) / (1 + x_exp)

    # compute the second derivative
    dy2dx2 = (a1 * a2**2 * x_exp) / ((1 + x_exp)**2)

    return y, dydx, dy2dx2
