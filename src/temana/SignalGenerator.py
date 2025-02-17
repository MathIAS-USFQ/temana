# External libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# Custom libraries
from .interpolators import mspline_t, msplin_zero

class SignalGenerator:
    """Generator of random signals of arbitrary resolution of variable amplitude and frecuency."""
    
    def __init__(self, start: float, end: float, num_points: int, sample_points: int = None):
        """Defines the interval for the X coordinates and the number of points for each generated signal.
        
        Parameters
        ----------
        start : float
            Initial point of the interval.
        end : float
            End point of the interval.
        num_points : int
            The number of points within the interval that will be generated for each signal.
        sample_points : int, optional
            The number of sampled points for a reduced domain. Must be less than `num_points`.
        """
        
        self.start = start
        self.end = end
        self.num_points = num_points
        
        self.add_sample = sample_points is not None
        if self.add_sample:
            if sample_points >= num_points:
                raise ValueError("sample_points must be strictly less than num_points.")
            
            self.sample_points = sample_points
        
        self._setup()

    def _setup(self):
        """Initializes the domain and the list of signals."""
        
        self._domain = np.linspace(self.start, self.end, self.num_points)
        self._signals = []
        
        if self.add_sample:
            self._sample_domain = np.linspace(self.start, self.end, self.sample_points)
            self._sampled_signals = []

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
    
    def _random_signal_function(self) -> callable:
        """Returns a sine or cosine function at random."""
        
        return np.sin if np.random.choice([0, 1], 1) == 0 else np.cos
    
    def _random_interpolator(self, xs, ys, tau) -> callable:
        """Returns a random interpolator function."""
        
        if np.random.choice([0,1], 1, p=[0.98, 0.02]) == 0:
             mspline_t(xs, ys, tau)
        return msplin_zero(xs, ys)

    def generate_signal(self):
        """Generate a random signal of arbitrary resolution of variable amplitude and frecuency.
        
        Returns
        -------
        y: np.ndarray
            Y coordinates of the signal at each point defined in `self.domain`.
        y_sampled: np.ndarray
            Y coordinates of the signal at each point defined in `self.sample_domain`. If `self.add_sample` is `False`, returns `None`.
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
        signal_function = self._random_signal_function()
        interpolator = self._random_interpolator(xs, ys, tau)
        
        def generate_points(x):
            y = A*signal_function(B(x)*(x - D))
            noise = noise_amplitude*mspline_t(xs, ys, tau)(x) * np.sin(msplin_zero(xs_h, ys_h)(x)*x)
            y = interpolator(x)*y + C + noise
            return y
        
        y = generate_points(self._domain)
        y_sampled = None
        
        if self.add_sample:
            y_sampled = generate_points(self._sample_domain)
        
        return y, y_sampled
    
    def generate_signals(self, num_signals: int = 1):
        """Generates a number of signals and stores them internally.
        
        Parameters
        ----------
        num_signals : int
            The number of signals to generate.
            
        Raises
        ------
        TypeError
            If `num_signals` is not an integer.
        ValueError
            If `num_signals` is less than 1.
        """
        
        try:
            if num_signals < 1:
                raise ValueError("The number of signals must be greater than 0.")
            
            for _ in range(num_signals):
                signal, sampled_signal = self.generate_signal()
                self._signals.append(signal)
                if self.add_sample:
                    self._sampled_signals.append(sampled_signal)
        except TypeError:
            raise TypeError("The number of signals must be an integer.")
            
    def get_signals(self) -> list:
        """Returns the list of generated signals."""
        
        return self._signals
    
    def get_sampled_signals(self) -> list:
        """Returns the list of sampled signals."""
        
        try:
            return self._sampled_signals
        except AttributeError:
            raise ValueError("No sampled signals have been generated")
    
    def plot_signal(self, signal_index: int = None, save_path: str = None):
        """Plots a signal from the list of generated signals.
        
        Parameters
        ----------
        signal_index : int
            The index of the signal to plot. If `None`, a random signal is chosen.
        save_path : str
            The path to save the plot to. If `None`, the plot is displayed.
            
        Raises
        ------
        ValueError
            If no signals have been generated yet.
        """

        if not self._signals:
            raise ValueError("No signals generated yet.")

        index = signal_index if signal_index is not None else random.randint(0, len(self._signals) - 1)
        signal = self._signals[index]

        plt.figure(figsize=(14, 7))
        # plt.style.use('ggplot')
        plt.plot(self._domain, signal, label="Signal")
        
        if self.add_sample:
            sampled_signal = self._sampled_signals[index]
            plt.scatter(self._sample_domain, sampled_signal, label="Sample Points", marker=".", color="crimson")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()
        
    def save_signals(self, filename: str, directory: str = "output", save_images: bool = False):
        """Saves the generated signals to a file in the specified directory.
        
        Parameters
        ----------
        filename : str
            The name of the file to save the signals to.
        directory : str
            The directory in which to save the file.
        save_images : bool
            Whether to save images of the signals as well.
            
        Raises
        ------
        ValueError
            If no signals have been generated yet.
        """
        
        if not self._signals:
            raise ValueError("No signals generated yet.")
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        self._save_signals_to_file(filename, directory)
        
        if self.add_sample:
            self._save_signals_to_file(f"sample_{filename}", directory, is_sampled=True)
            
        if save_images:
            images_dir = os.path.join(directory, "imgs")
            self._save_signals_images(images_dir)

    def _save_signals_to_file(self, filename: str, directory: str, is_sampled: bool = False):
        """Helper method to save the generated signals to a file.
        
        Parameters
        ----------
        filename : str
            The name of the file to save the signals to.
        directory : str
            The directory in which to save the file.
        is_sampled : bool
            Whether the signals are sampled or not.
        """
        
        filepath = os.path.join(directory, filename)
        with open(filepath, "w") as f:
            for i, signal in enumerate(self._signals):
                f.write(" ".join(map(str, signal)))
                if i != (len(self._signals) - 1):
                    f.write("\n")
                print(f'Saved {"sampled " if is_sampled  else ""}signal {i + 1}/{len(self._signals)}', end="\r")
        
        cursive_filepath = f'\x1B[3m{filepath}\x1B[0m'
        if is_sampled:
            print(f'Sampled signals saved to {cursive_filepath}.')
        else:
            print(f"Signals saved to {cursive_filepath}.")
            
    def _save_signals_images(self, images_dir: str):
        """Helper method to save images of the generated signals.
        
        Parameters
        ----------
        images_dir : str
            The directory in which to save the images.
        """
        
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        for i in range(len(self._signals)):
            image_path = os.path.join(images_dir, f"{i}.png")
            self.plot_signal(signal_index=i, save_path=image_path)
            print(f"Saved signal image {i + 1}/{len(self._signals)}", end="\r")
        
        cursive_images_dir = f'\x1B[3m{images_dir}\x1B[0m'
        print(f"Images saved to {cursive_images_dir} directory.")