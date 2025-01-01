import numpy as np
from scipy.interpolate import interp1d, make_interp_spline

def mspline_t(x: np.ndarray, y: np.ndarray, tau : float):
    """Generate a tension spline interpolation as described in [Numerical Analysis by David Kincaid](https://books.google.com.ec/books?id=kPDtAp3UZtIC&source=gbs_navlinks_s) and [here](https://catxmai.github.io/pdfs/Math212_ProjectReport.pdf).
    
    Parameters
    ----------
    x : np.ndarray
        Points at which the function changes its character (aka knots).
    y : np.ndarray
        Values of the function at each knot.
    tau : float
        Tension parameter that generates different conditions for the spline. For tau = 0 we recover the cubic spline, while higher values of tau make the function resemble a linear spline.
        
    Returns
    -------
    callable
        Interpolation function defined by tension spline approximation that can be evaluated at several points at once.
    """
    n = len(x) - 1
    
    # Step 1: Calculate hi, alpha, beta, gamma
    h = np.diff(x)
    alpha = 1/h - tau/np.sinh(tau*h)
    beta = tau*np.cosh(tau*h)/np.sinh(tau*h) - 1/h
    gamma = tau**2 * np.diff(y)/h

    # Step 2: Set up the tridiagonal system A * Z = Y
    A = np.zeros((n+1, n+1))
    Y = np.zeros(n+1)
    
    A[0, 0] = 1
    A[n, n] = 1
    for i in range(1, n):
        A[i, i-1] = alpha[i-1]
        A[i, i] = beta[i-1] + beta[i]
        A[i, i+1] = alpha[i]
        Y[i] = gamma[i] - gamma[i-1]
    
    # Solve the system for Z
    Z = np.linalg.solve(A, Y)
    
    # Step 3: Define the tension spline function
    def tension_spline(x_eval : np.ndarray) -> callable:
        result = np.zeros_like(x_eval)
        for i in range(n):
            mask = (x_eval >= x[i]) & (x_eval <= x[i+1])
            xi = x_eval[mask]
            t1 = Z[i]*np.sinh(tau*(x[i+1] - xi)) + Z[i+1]*np.sinh(tau*(xi - x[i]))
            t1 /= tau**2 * np.sinh(tau*h[i])
            t2 = (y[i] - Z[i]/tau**2) * (x[i+1] - xi)/h[i]
            t3 = (y[i+1] - Z[i+1]/tau**2) * (xi - x[i])/h[i]
            result[mask] = t1 + t2 + t3
        return result
    
    return tension_spline

def msplin_zero(x: np.ndarray, y: np.ndarray) -> callable:
    """Return a interpolation function of degree zero."""

    return interp1d(x, y, kind='zero', fill_value="extrapolate")

def msplin_one(x: np.ndarray, y: np.ndarray) -> callable:
    """Return a interpolation function of degree one."""

    return interp1d(x, y, kind='linear', fill_value="extrapolate")

def mbsplin_n(x: np.ndarray, y: np.ndarray, n) -> callable:
    """Return a interpolation function of B-splines of degree n."""
    return make_interp_spline(x, y, k=n)