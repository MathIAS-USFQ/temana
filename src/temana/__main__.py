from temana.args_parser import parse_args
from temana import SignalGenerator


def main():
    args = parse_args()

    # Initialize the signal generator
    generator = SignalGenerator(start=args.start, end=args.end, num_points=args.points)

    # Generate signals
    generator.generate_signals(num_signals=args.signals)

    # Save signals to the specified file and directory
    generator.save_signals(
        filename=args.file, directory=args.directory, save_images=args.imgs
    )


if __name__ == "__main__":
    main()
