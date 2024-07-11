from aifinder.depth_finder import DepthFinder
import signal
import sys


def main() -> None:
    # Initialize the depth camera and the Yolo model
    depth_finder = DepthFinder(640, 480, 30, 'yolov8s.pt')

    # This function allows to terminate the program gracefully when doing Ctrl+C
    def terminate(*_):
        print("Terminating gracefully...")
        depth_finder.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, terminate)

    while True:
        # Update the camera frames
        depth_finder.update(iou = 0.7)

        if depth_finder.visible_objects is not None and len(depth_finder.visible_objects) > 0:
            depth_finder.get_edges_of_object(0)

        # Query a black mouse (can be any object available in the chosen model)
        print(depth_finder.find_object_by_name_and_color('mouse', 'black'))


if __name__ == '__main__':
    main()
