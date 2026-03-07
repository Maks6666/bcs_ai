import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Military units tracker')
    parser.add_argument('--path', type=str, required=True, help='Path to video')
    parser.add_argument('--atgm', type=int, default=30, help='Amount of ATGM')
    parser.add_argument('--cl_shells', type=int, default=30, help='Amount of cluster shells')
    parser.add_argument('--un_shells', type=int, default=30, help='Amount of unitary shells')
    parser.add_argument('--fpv', type=int, default=30, help='Amount of FPV frones')

    return parser.parse_args()