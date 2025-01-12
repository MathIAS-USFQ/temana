import os
import pytest
from temana import SignalGenerator

# Helper function to generate a sample generator with default params
def create_default_generator(sample_points=None):
    return SignalGenerator(0, 10, 1000, sample_points=sample_points)

def test_generate_signals():
    generator = create_default_generator()
    generator.generate_signals(5)
    assert len(generator.get_signals()) == 5

def test_generate_sampled_signals():
    generator = create_default_generator(sample_points=100)
    generator.generate_signals(3)
    assert len(generator.get_sampled_signals()) == 3

@pytest.mark.parametrize("sample_points", [None, 100])
def test_sampling_logic(sample_points):
    generator = create_default_generator(sample_points=sample_points)
    signal, sampled_signal = generator.generate_signal()

    assert len(signal) == 1000
    if sample_points:
        assert len(sampled_signal) == sample_points
    else:
        assert sampled_signal is None
        
def test_save_signals(tmp_path):
    generator = create_default_generator()
    generator.generate_signals(3)

    file_path = tmp_path / "signals.txt"
    generator.save_signals(filename=str(file_path))

    assert os.path.exists(file_path)

    with open(file_path, "r") as f:
        lines = f.readlines()

    assert len(lines) == 3

def test_save_sampled_signals(tmp_path):
    generator = create_default_generator(sample_points=100)
    generator.generate_signals(2)

    filename = "signals.txt"
    sample_file_path = f"sample_{filename}"
    generator.save_signals(filename=filename, directory=str(tmp_path))
    
    sample_file_path = os.path.join(tmp_path, sample_file_path)
    assert os.path.exists(sample_file_path)

    with open(sample_file_path, "r") as f:
        lines = f.readlines()

    assert len(lines) == 2

def test_save_images(tmp_path):
    generator = create_default_generator()
    generator.generate_signals(2)

    output_dir = tmp_path / "output"
    generator.save_signals(
        filename="signals.txt", directory=str(output_dir), save_images=True
    )

    images_dir = output_dir / "imgs"
    assert os.path.exists(images_dir)
    assert len(os.listdir(images_dir)) == 2

def test_save_sampled_images(tmp_path):
    generator = create_default_generator(sample_points=100)
    generator.generate_signals(2)

    output_dir = tmp_path / "output"
    generator.save_signals(
        filename="signals.txt", directory=str(output_dir), save_images=True
    )

    images_dir = output_dir / "imgs"
    assert os.path.exists(images_dir)
    assert len(os.listdir(images_dir)) == 2

def test_plot_signal(tmp_path):
    generator = create_default_generator()
    generator.generate_signals(1)

    image_path = tmp_path / "signal_plot.png"
    generator.plot_signal(signal_index=0, save_path=str(image_path))

    assert os.path.exists(image_path)

def test_invalid_sample_points():
    with pytest.raises(ValueError):
        SignalGenerator(0, 10, 1000, sample_points=1000)

    with pytest.raises(ValueError):
        SignalGenerator(0, 10, 1000, sample_points=-1)

    with pytest.raises(TypeError):
        SignalGenerator(0, 10, 1000, sample_points="invalid")
