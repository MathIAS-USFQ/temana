import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pacakge to generate signals of random amplitude and frequency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Name of the file to save the signals into",
    )
    parser.add_argument(
        "--start", "-s", type=int, default=0, help="Start value of the signal domain"
    )
    parser.add_argument(
        "--end", "-e", type=int, default=1, help="End value of the signal domain"
    )
    parser.add_argument(
        "--signals", "-n", type=int, default=1, help="Number of signals to be generated"
    )
    parser.add_argument(
        "--points", "-p", type=int, default=100, help="Number of points of each signal"
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default="output",
        help="Directory to save the files",
    )
    parser.add_argument(
        "--imgs", "-i", action="store_true", help="Save signal plots as images"
    )

    return parser.parse_args()
