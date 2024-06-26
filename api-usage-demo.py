from depth_finder import DepthFinder
import signal
import sys

def main() -> None:
    depth_finder = DepthFinder(640, 480, 30, 'yolov8s.pt')

    def terminate(*_):
        print("Terminating gracefully...")
        depth_finder.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, terminate)

    while True:
        depth_finder.update()

        print(depth_finder.find_object_by_name_and_color('water-bottle', 'black'))

if __name__ == '__main__':
    main()
