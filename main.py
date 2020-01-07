import argparse

from src.supervisor import Supervisor


parser = argparse.ArgumentParser(description='Mosse object tracking')
parser.add_argument('-f', '--file', type=str, required=False,
                    help='Path to video file. If not specified, it will use the camera')
parser.add_argument('-w', '--width', type=int, required=False,
                    help='Width of displayed video')
args = parser.parse_args()


def main():
    port = args.file if args.file else 0
    tracker = Supervisor(port)
    tracker.run()


if __name__ == '__main__':
    main()
