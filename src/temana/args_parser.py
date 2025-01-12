import argparse
import numpy as np

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
        "--start", "-s", type=float, default=0, help="Start value of the signal domain"
    )
    parser.add_argument(
        "--end", "-e", type=float, default=4*np.pi, help="End value of the signal domain"
    )
    parser.add_argument(
        "--signals", "-n", type=int, default=1, help="Number of signals to be generated"
    )
    parser.add_argument(
        "--points", "-p", type=int, default=1000, help="Number of points of each signal"
    )
    parser.add_argument(
        "--sample_points",
        type=int,
        help="Number of points for the sample signal. Must be strictly less than the number of points of the signal",
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
