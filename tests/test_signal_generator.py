import os
from temana import SignalGenerator

def test_generate_signal():
    generator = SignalGenerator(0, 10, 1000)
    signal = generator.generate_signal()
    assert len(signal) == 1000

def test_generate_signals():
    generator = SignalGenerator(0, 10, 1000)
    generator.generate_signals(5)
    assert len(generator.get_signals()) == 5

def test_save_signals(tmp_path):
    generator = SignalGenerator(0, 10, 1000)
    generator.generate_signals(3)

    file_path = tmp_path / "signals.txt"
    generator.save_signals(filename=str(file_path))

    assert os.path.exists(file_path)

    with open(file_path, "r") as f:
        lines = f.readlines()

    assert len(lines) == 3

def test_save_images(tmp_path):
    generator = SignalGenerator(0, 10, 1000)
    generator.generate_signals(2)

    output_dir = tmp_path / "output"
    generator.save_signals(
        filename="signals.txt", directory=str(output_dir), save_images=True
    )

    images_dir = output_dir / "imgs"
    assert os.path.exists(images_dir)
    assert len(os.listdir(images_dir)) == 2