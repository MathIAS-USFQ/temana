# temana

`temana` is a Python package for generating random signals of variable amplitude and frequency. The package allows you to generate signals with as many points as needed in a specified domain and provides functionality to save the signals to a file or as images.

![Example signal](https://github.com/MathIAS-USFQ/temana/blob/main/assets/example.png?raw=true)

---

## Installation

You can install `temana` using:

```bash
pip install temana
```

---

## Usage

To use the `temana` package, you can run it as a script or import it into your Python code.

### **Command-Line Usage**

You can generate and save signals from the command line using the package's main entry point:

```bash
temana -f signals.txt
```

The signals file then can be found in `./output/signals.txt`.

**Arguments:**
- `--file, -f`: Name of the file to save the generated signals. At the moment the package only supports `.txt` files. 
- `--start, -s`: Start value of the signal domain. Defaults to 0.
- `--end, -e`: End value of the signal domain. Defaults to 4$\pi$.
- `--signals, -n`: Number of signals to generate. Defaults to 1.
- `--n_points, -p`: Number of points in each signal. Defaults to 1000.
- `--directory, -d`: Directory to save the files. Defaults to `./output`.
- `--imgs, -i`: Save signal plots as images if used.
- `--sample_points`: Number of points to sample from the original signal. Must be stictly less than `--points`. The sample signals will always be saved to `sample_{filename}`, where `filename` is the name specified in the `-f` parameter. If not specified, only the full signals will be generated.

### **Using the Package in Python**

```python
from temana import SignalGenerator

# Create a SignalGenerator instance
generator = SignalGenerator(start=0, end=10, num_points=1000)

# Generate 5 signals
generator.generate_signals(num_signals=5)

# Save the signals to a file and images
generator.save_signals("signals.txt", save_images=True)
```

---

## Development

To contribute to `temana`, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/MathIAS-USFQ/temana.git
   ```

1. Install the package in editable mode along with development dependencies:
   ```bash
   pip install -e .[dev]
   ```

---

## Tests

The package uses `pytest` for testing. To run the tests use the command:

```bash
pytest
```

---

## Authors

- **Julio Ibarra** - [jibarra@usfq.edu.ec](mailto:jibarra@usfq.edu.ec)
- **José Ocampo** - [joseocampo220@gmail.com](mailto:joseocampo220@gmail.com)

---

## License

This project is licensed under the **MIT License**.  

