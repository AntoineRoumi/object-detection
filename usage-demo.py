from depth_finder import DepthFinder
import signal
import sys

def main() -> None:
    depth_finder = DepthFinder(640, 480, 30, './bluecups.pt')

    def terminate(*_):
        print("Terminating gracefully...")
        depth_finder.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, terminate)

    print(depth_finder.model.classes_ids)

    while True:
        depth_finder.update()

        print(depth_finder.find_object_by_name('water-bottle'))

if __name__ == '__main__':
    main()
