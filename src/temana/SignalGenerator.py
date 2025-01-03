import numpy as np
import random

from interpolators import mspline_t, msplin_zero

class SignalGenerator:
    """Generator of random signals of arbitrary resolution of variable amplitude and frecuency."""
    
    def __init__(self, start: float, end: float, num_points: int):
        """Defines the interval for the X coordinates and the number of points for each generated signal.
        
        Parameters
        ----------
        start : float
            Initial point of the interval.
        end : float
            End point of the interval.
        num_points : int
            The number of points within the interval that will be generated for each signal.
        """
        
        self.start = start
        self.end = end
        self.num_points = num_points
        
        self._setup()

    def _setup(self):
        """Initializes the domain and the list of signals."""
        
        self._domain = np.linspace(self.start, self.end, self.num_points)
        self._signals = []

    def _random_signal_function(self) -> callable:
        """Returns a sine or cosine function at random."""
        
        return np.sin if np.random.choice([0, 1], 1) == 0 else np.cos

    def _amplitude_change_points(self) -> list:
        """Create a list of points in the domain [`self.start`, `self.end`] to change the amplitude of the signal.
        
        Returns
        -------
        points : list
            List of points where the X-coordinate is the point the interval and the Y-coordinate the amplitude value.
        """
        
        A = (2*np.random.rand() - 1)*np.random.randint(1, 10, 1) #Amplitude
        B = (2*np.random.rand() - 1)*np.random.randint(1, 10, 1) #Frequency
        C = (np.random.rand())*np.pi #Phase
            
        num_points = random.randint(1, 4)
        
        partition = np.linspace(self.start, self.end, num_points + 2)
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

    def _freq_change_points(self):
        """Create a list of points in the domain [`self.start`, `self.end`] to change the frecuency of the signal.
            
        Returns
        -------
        points : list
            List of points where the X-coordinate is the point the interval and the Y-coordinate the frequency value. 
        points_h : list
            List of points where the X-coordinate is the point the interval and the Y-coordinate the frequency value, including high frequencies. 
        """
            
        #Create lists of high and low frequencies 
        high_frequencies = np.sort(np.random.uniform(20, 100, 10))
        low_frequencies = np.sort(np.random.uniform(1, 5, 10))
        
        #Choose randomly the number of points for the change of frequency
        num_points = random.randint(2, 11)
        
        #Create a random partition of the interval [self.start, self.end]
        domain = np.zeros(num_points + 2) #The +2 is to take into account start and end points
        domain[-1] = 1
        domain[1:-1] = np.sort(np.random.rand(num_points)) 
        partition = self.start + (self.end - self.start)*domain
        
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
        
        return points, points_h

    def generate_signal(self):
        """Generate a random signal of arbitrary resolution of variable amplitude and frecuency.
        
        Returns
        -------
        y: np.ndarray
            Y coordinates of the signal at each point defined in self.domain.
        """
        
        freq_points, freq_points_h = self._freq_change_points()
        xf, yf = zip(*freq_points)
        tau = np.random.choice(np.linspace(1, 2, 11))

        A = (2*np.random.rand() - 1)*np.random.randint(1, 6, 1) #Amplitude
        B = mspline_t(xf, yf, tau) #Frecuency
        C = (2*np.random.rand()-1)*np.random.randint(1, 10, 1) #Translation
        D = (np.random.rand())*np.pi #Phase
        
        amp_points = self._amplitude_change_points()
        xs, ys = zip(*amp_points)
        xs_h, ys_h = zip(*freq_points_h)
        
        tau = np.random.choice(np.linspace(1, 2, 21))
        noise_amplitude = np.random.uniform(0.08, 0.2, 1)
        signal_funtion = self._random_signal_function()
        
        x = self._domain
        y = A*signal_funtion(B(x)*(x - D))
        noise = noise_amplitude*mspline_t(xs, ys, tau)(x) * np.sin(msplin_zero(xs_h, ys_h)(x)*x)
        
        if np.random.choice([0,1], 1, p=[0.98, 0.02]) == 0: #Splines of random tension
            y = mspline_t(xs, ys, tau)(x)*y + C + noise
        else: #Splines of infite tension
            y = msplin_zero(xs, ys)(x)*y + C + noise

        return y