from tracker import Tracker
from parser import parse_args

def main():
    args = parse_args()
    weapons = {'atgm': args.atgm, 'cluster_shells': args.cl_shells, 'unitary_shells': args.un_shells, 'fpv_drones': args.fpv}
    tracker = Tracker(path = args.path, weapons=weapons)
    tracker()

if __name__ == "__main__":
    main()