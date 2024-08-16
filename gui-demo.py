#!/usr/bin/python3

from aifinder.camera import DepthCamera
from aifinder.model import BoundingBox, YoloModel
from aifinder import gui
from aifinder import gpu_utils
from aifinder import image_manipulation as imanip
from aifinder import color_recognition
from aifinder import edge_detection as ed

import cv2
import glfw

# Characteristics of the depth camera
WIDTH, HEIGHT = 1280, 720
FPS = 30

# ImGui windows size
RESULTS_WINDOW_W, RESULTS_WINDOW_H = 180, 200
METRICS_WINDOW_W, METRICS_WINDOW_H = 180, 80
MOUSE_COORDS_W, MOUSE_COORDS_H = 180, 60
SLIDERS_WINDOW_W, SLIDERS_WINDOW_H = 180, 80

# The training dataset for the color recognition algorithm
TRAINING_DATA_DIR = './training_dataset'
TRAINING_DATA_FILE = './training.data'


def main():
    # Initialization of the depth camera
    camera = DepthCamera(width=WIDTH, height=HEIGHT, fps=30)

    # Initialization of the yolo model
    model = YoloModel('./bluecups.pt')

    # Initialization of the GUI window
    window = gui.Window("Yolov8", WIDTH, HEIGHT)

    # Initialization of the ImGui windows
    results_window = gui.ImguiTextWindow('Results', 10, 30, RESULTS_WINDOW_W,
                                         RESULTS_WINDOW_H)
    mouse_coords_window = gui.ImguiTextWindow('Mouse coords', 10, results_window.y + RESULTS_WINDOW_H + 10,
                                              MOUSE_COORDS_W, MOUSE_COORDS_H)
    metrics_window = gui.ImguiTextWindow(
        'Metrics', 10, mouse_coords_window.y + MOUSE_COORDS_H + 10,
        METRICS_WINDOW_W, METRICS_WINDOW_H)

    # Initialization of computer usage metrics
    t0 = glfw.get_time()
    t = 0
    fps = window.get_fps()
    gpu_total_mem = gpu_utils.query_gpu_total_mem()
    gpu_used_mem = gpu_utils.query_gpu_used_mem(units=False)
    gpu_util = gpu_utils.query_gpu_utilization()
    metrics_window.set_text(
        f"FPS: {fps}\nGPU Usage: {gpu_utils.query_gpu_utilization()}%\nGPU mem: {gpu_utils.query_gpu_used_mem(units=False)}/{gpu_total_mem}"
    )

    # Initialization of Yolo inference parameters (is_sliders_expand and is_sliders_close are used for the ImGui window with the parameters sliders)
    iou_thres = 0.5
    conf_thres = 0.8
    canny_low = 100
    canny_high = 200
    is_sliders_expand = True
    is_sliders_close = True

    # Initialization of the text of ImGui windows
    results_str = []
    color_prediction = ''
    test_histogram = ''

    coords_under_cursor = None
    distance_under_cursor = None
    coords_under_cursor_text = ''
    cursor_x, cursor_y = 0., 0.

    edges = None

    # Training of the color recognition model
    print("training color recognition")
    color_recognition.training(TRAINING_DATA_DIR, TRAINING_DATA_FILE)
    color_classifier = color_recognition.KnnClassifier("./training.data")
    print("trained color recognition")

    while not window.should_close():
        # Update of the camera frames
        camera.update_frame()
        color_frame = camera.get_color_frame_as_ndarray()
        if color_frame is None:
            continue
        # Run a prediction on the updated color frame
        results = model.predict_frame(color_frame,
                                      iou=iou_thres,
                                      conf=conf_thres)

        results_frame = results.render()

        # Process of each detected object in the results
        results_str = []
        for i in range(results.results_count()):
            # Get the coordinates of the object
            bbox = results.get_box_coords(i)
            coords, center_distance = camera.get_coords_of_object_xyxy(bbox)
            # Prediction of the color of the object
            quarter_width = (bbox.x1 - bbox.x0) // 4
            quarter_height = (bbox.y1 - bbox.y0) // 4
            inner_image = BoundingBox(bbox.x0 + quarter_width,
                                      bbox.y0 + quarter_height,
                                      bbox.x1 - quarter_width,
                                      bbox.y1 - quarter_height)

            test_histogram = color_recognition.color_histogram_of_image(imanip.extract_area_from_image(color_frame, inner_image))
            color_prediction = color_classifier.predict(test_histogram)
            # Format results for display
            if center_distance is None or coords is None:
                results_str.append(
                    f"{results.get_class_name(i)} ({results.get_conf(i):.2f}):\n\tnot in range\n\tcolor: {color_prediction}\n"
                )
            else:
                results_str.append(
                    f"{results.get_class_name(i)} ({results.get_conf(i):.2f}):\n\t{center_distance:.3f}mm\n\tcolor: {color_prediction}\n\t({coords.x:.1f},{coords.y:.1f},{coords.z:.1f})"
                )
            
            edges = ed.edge_detection_rectangle_on_frame(color_frame, bbox, canny_low, canny_high)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            results_frame[bbox.y0:bbox.y1, bbox.x0:bbox.x1] = edges

        results_window.set_text('\n'.join(results_str))

        # Calculate GPU usage
        t = glfw.get_time()
        fps = window.get_fps()
        if t - t0 > 1.0:
            t0 = t
            gpu_util = gpu_utils.query_gpu_utilization()
            gpu_used_mem = gpu_utils.query_gpu_used_mem(units=False)
        metrics_window.set_text(
            f"FPS: {fps:.1f}\nGPU Usage: {gpu_util}%\nGPU mem: {gpu_used_mem}/{gpu_total_mem}"
        )

        # GUI stuff
        window.begin_drawing()

        # Render the prediction image
        window.draw_background_from_mem(results_frame, WIDTH, HEIGHT)

        gui.im.set_next_window_position(10,
                                        metrics_window.y + METRICS_WINDOW_H +
                                        10,
                                        condition=gui.im.ONCE)
        gui.im.set_next_window_size(SLIDERS_WINDOW_W,
                                    SLIDERS_WINDOW_H,
                                    condition=gui.im.ONCE)
        if is_sliders_close:
            is_sliders_expand, is_sliders_close = gui.im.begin(
                "Prediction parameters")
            if is_sliders_expand:
                _, iou_thres = gui.im.slider_float("iou",
                                                   iou_thres,
                                                   min_value=0.0,
                                                   max_value=0.7,
                                                   format="%.2f")
                _, conf_thres = gui.im.slider_float("conf",
                                                    conf_thres,
                                                    min_value=0.0,
                                                    max_value=1.0,
                                                    format="%.2f")
                _, canny_low = gui.im.slider_int("canny_low",
                                                 canny_low,
                                                 min_value=50,
                                                 max_value=canny_high-1)
                _, canny_high = gui.im.slider_int("canny_high",
                                                  canny_high,
                                                  min_value=canny_low+1,
                                                  max_value=800)
            gui.im.end()

        cursor_x, cursor_y = window.get_cursor_pos_in_window()
        if cursor_x is None or cursor_y is None:
            coords_under_cursor_text = 'Cursor out of window'
        else:
            coords_under_cursor, distance_under_cursor = camera.get_coords_of_pixel(int(cursor_x), int(cursor_y))
            if coords_under_cursor is None or distance_under_cursor is None:
                coords_under_cursor_text = 'No depth data'
            else:
                coords_under_cursor_text = f'{coords_under_cursor.x:.2f}, {coords_under_cursor.y:.2f}, {coords_under_cursor.z:.2f}\n{distance_under_cursor}mm'

        mouse_coords_window.set_text(coords_under_cursor_text)

        window.draw_imgui_text_window(results_window)
        window.draw_imgui_text_window(mouse_coords_window)
        window.draw_imgui_text_window(metrics_window)

        window.end_drawing()

    print('Terminating...')
    window.terminate()
    camera.terminate()


if __name__ == '__main__':
    main()
