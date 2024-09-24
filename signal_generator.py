"""
Signal generator is the library of functions for temporal data analysis and generation of artificial signals.
"""

import numpy as np
from scipy.interpolate import interp1d, make_interp_spline
import random
from numpy.typing import ArrayLike
                    
def random_signal_function() -> callable:
    """Returns at random a sine or cosine function."""
    return np.sin if np.random.choice([0, 1], 1) == 0 else np.cos
        
def generate_signal(start: float, end: float, num_points: ArrayLike):
    """Generate random signals of arbitrary resolution of variable frecuency.
    
    Parameters
    ----------
    start : int
        The start of the domian interval, inclusive.
    end : int
        Then end of the domain interval, inclusive.
    num_points : list
        The number of points for each generated signal. Must be non-empty.
    add_noise : bool
        Whether to add noise to the signal or not.
        
    Returns
    -------
    domains : list
        X coordinates of each signal.
    signals: list
        Y coordinates of the each signal.
    """
    
    A = (2*np.random.rand() - 1)*np.random.randint(1, 6, 1) #Amplitude
    B = (2*np.random.rand() - 1)*np.random.randint(1, 10, 1) #Frecuency
    C = (2*np.random.rand() - 1)*np.random.randint(1, 10, 1) #Translation
    signal_function = random_signal_function()

    noise_amp = (2*np.random.rand() - 1)*np.random.uniform(0.1, 0.5, 1) #Noise amplitude
    noise_freq = (2*np.random.rand() - 1)*np.random.randint(100, 200, 1) #Noise frecuency
    noise_function = random_signal_function()

    domains = []
    signals = []
    
    for n in num_points:
        x = np.linspace(start, end, n)
        y = A*signal_function(B*x) + C
        y += noise_amp*noise_function(noise_freq*x);
        
        domains.append(x)
        signals.append(y)
        
    return domains, signals
    
##########################################################################################
#Interpolator functions
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

##########################################################################################

def amplitude_change_points(start: float, end: float) -> list:
    """Generate a list between 1 and 4 reference points in the interval [start, end] where the amplitud changes."""
    
    A = (2*np.random.rand() - 1)*np.random.randint(1, 10, 1) #Amplitude
    B = (2*np.random.rand() - 1)*np.random.randint(1, 10, 1) #Frequency
    C = (np.random.rand())*np.pi #Phase
        
    num_points = random.randint(1, 4)
    
    partition = np.linspace(start, end, num_points + 2)
    points = []
    
    for i in range(num_points + 2):
        x = partition[i]
        y = (A*np.sin(B*(x - C)))[0]
        y = np.abs(y)
        #Verify the amplitude isn't too low
        if y < 0.5:
            y = (A*np.random.uniform(0.5, 1))[0]
            y = np.abs(y)
        points.append((x, y))
    
    return points

def generate_signal_va(start: float, end: float, num_points: ArrayLike):
    """Generate random signals of arbitrary resolution of variable frecuency.
    
    Parameters
    ----------
    start : int
        The start of the domian interval, inclusive.
    end : int
        Then end of the domain interval, inclusive.
    num_points : array_like
        The number of points for each generated signal. Must be non-empty.
        
    Returns
    -------
    domains : list
        X coordinates of each signal.
    signals: list
        Y coordinates of the each signal.
    """
    
    A = (2*np.random.rand() - 1)*np.random.randint(1, 6, 1) #Amplitude
    B = (2*np.random.rand() - 1)*np.random.randint(1, 25, 1) #Frecuency
    C = (2*np.random.rand() - 1)*np.random.randint(1, 10, 1) #Translation
    D = (np.random.rand())*np.pi #Phase
    
    domains = []
    signals = []
    
    points = amplitude_change_points(start, end)
    xs, ys = zip(*points)
    tau = np.random.choice([1, 3, 5, 8, 10, 12, 15, 20])
    
    signal_function = random_signal_function()
    
    for n in num_points:
        x = np.linspace(start, end, n)
        y = A*signal_function(B*(x - D))
    
        if np.random.randint(0, 2, 1) == 0: #Splines of random tension
            y = mspline_t(xs, ys, tau)(x)*y + C
        else: #Splines of infite tension
            y = msplin_zero(xs, ys)(x)*y + C

        domains.append(x)
        signals.append(y)
        
    return domains, signals 

def freq_change_points(start: float, end: float):
    """Create a list of points in the domain [start, end] to change the frecuency of the signal.
    
    Parameters
    ----------
    start : float
        Initial point of the interval.
    end : float
        End point of the interval.
        
    Returns
    -------
    points : list
        List of points where the X-coordinate is the point the interval and the Y-coordinate the frequency value. 
    points_h : list
        List of points where the X-coordinate is the point the interval and the Y-coordinate the frequency value, including high frequencies. 
    type : list
        List of the frecuency type of each point.
    """
        
    #Create lists of high and low frequencies 
    high_frequencies = np.sort(np.random.uniform(20, 100, 10))
    low_frequencies = np.sort(np.random.uniform(1, 5, 10))
    
    #Choose randomly the number of points for the change of frequency
    num_points = random.randint(2, 11)
    
    #Create a random partition of the interval [start, end]
    domain = np.zeros(num_points + 2) #The +2 is to take into account start and end points
    domain[-1] = 1
    domain[1:-1] = np.sort(np.random.rand(num_points)) 
    partition = start + (end - start)*domain
    
    points = [] #Stores points for the change of frecuency
    f_types = [] #Stores the frecuency type of each point
    high_f_domain = [domain[0]] # Include information about high frecuencies only
    high_frecuencies = [] #Stores frencuency values when it changes to high
    
    #Initialize variation type
    variation_type = np.random.choice(["low", "high"], p=[0.96, 0.04])
    
    def handle_frequency_variation(type: str, i: int):
        """Stores frequency values according to their type."""
        nonlocal high_f_domain, high_frecuencies
            
        match type:
            case 'low' | 'no_change':
                high_frecuencies.append(0)
                
            case 'high':
                #Choose a random coordinate in the partition and select a high frecuency to be associated with that coordinate
                if i != num_points+1:
                    high_f_domain.append(np.random.uniform(partition[i], partition[i+1]))
                    high_frecuency = np.random.choice(high_frequencies)
                    high_frecuencies.append(high_frecuency)
                    high_frecuencies.append(high_frecuency)
                else:
                    high_frecuencies.append(0)
                    
    #Handle the first point in the interval
    f_types.append(variation_type)
    handle_frequency_variation(variation_type, 0)
    frequency = np.random.choice(low_frequencies) #Start with low frequency
    points.append((partition[0], frequency))
    
    #Handle rest of points
    for i in range(1, num_points + 2):
        x = partition[i]
        high_f_domain.append(x)

        if f_types[-1] == 'high':
            choice_params = {'a': ["low", "no_change"], 'p': [0.95, 0.05]} 
        else:
            choice_params = {'a': ["low", "high", "no_change"], 'p': [0.20, 0.07, 0.73]}
            
        variation_type = np.random.choice(**choice_params)
        handle_frequency_variation(variation_type, i)
        
        if variation_type == 'no_change':
            f_types.append(f_types[-1])
        else:
            frequency = np.random.choice(low_frequencies) #Force a frecuency change to low if variation_type is high or low
            f_types.append(variation_type)

        points.append((x, frequency))
        
    points_h = [(high_f_domain[i], high_frecuencies[i]) for i in range(len(high_f_domain))]
    
    return points, points_h, f_types

def generate_signal_va_vf(start: float, end: float, num_points: ArrayLike):
    """Generate random signals of arbitrary resolution of variable amplitude and frecuency.
    Parameters
    ----------
    start : int
        The start of the domian interval, inclusive.
    end : int
        Then end of the domain interval, inclusive.
    num_points : array_like
        The number of points for each generated signal. Must be non-empty. #TODO add validation to check for non-emptiness.
        
    Returns
    -------
    domains : list
        X coordinates of each signal.
    signals: list
        Y coordinates of the each signal.
    signal_noise : list
        Noise that was added to each signal. 
    """
    
    freq_points, freq_points_h, f_types = freq_change_points(start, end)
    xf, yf = zip(*freq_points)
    tau = np.random.choice(np.linspace(1, 2, 11))

    A = (2*np.random.rand() - 1)*np.random.randint(1, 6, 1) #Amplitude
    B = mspline_t(xf, yf, tau) #Frecuency
    C = (2*np.random.rand()-1)*np.random.randint(1, 10, 1) #Translation
    D = (np.random.rand())*np.pi #Phase
    
    domains = []
    signals = []
    signals_noise = []
    
    amp_points = amplitude_change_points(start, end)
    xs, ys = zip(*amp_points)
    xs_h, ys_h = zip(*freq_points_h)
    
    tau = np.random.choice(np.linspace(1, 2, 21))
    noise_amplitude = np.random.uniform(0.08, 0.2, 1)
    signal_funtion = random_signal_function()
    
    for n in num_points:
        x = np.linspace(start, end, n)
        y = A*signal_funtion(B(x)*(x - D))
        noise = noise_amplitude*mspline_t(xs, ys, tau)(x) * np.sin(msplin_zero(xs_h, ys_h)(x)*x)
        
        if np.random.choice([0,1], 1, p=[0.98, 0.02]) == 0: #Splines of random tension
            y = mspline_t(xs, ys, tau)(x)*y + C + noise
        else: #Splines of infite tension
            y = msplin_zero(xs, ys)(x)*y + C + noise
            
        domains.append(x)
        signals.append(y)
        signals_noise.append(noise)

    return domains, signals, signals_noise