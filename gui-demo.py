from camera import DepthCamera
from model import BoundingBox, YoloModel
import gui
import gpu_utils
import glfw
import image_manipulation as imanip
import color_recognition
import cv2
import OpenGL.GL as gl
import edge_detection as ed

# Characteristics of the depth camera
WIDTH, HEIGHT = 1280, 720
FPS = 30

# ImGui windows size
RESULTS_WINDOW_W, RESULTS_WINDOW_H = 180, 200
METRICS_WINDOW_W, METRICS_WINDOW_H = 180, 80
SLIDERS_WINDOW_W, SLIDERS_WINDOW_H = 180, 80

# The training dataset for the color recognition algorithm
TRAINING_DATA_DIR = './training_dataset'
TRAINING_DATA_FILE = './training.data'


def main():
    # Initialization of the depth camera
    camera = DepthCamera(width=WIDTH, height=HEIGHT, fps=30)

    # Initialization of the yolo model
    model = YoloModel('yolov8s.pt')

    # Initialization of the GUI window
    window = gui.Window("Yolov8", WIDTH, HEIGHT)

    # Initialization of the ImGui windows
    results_window = gui.ImguiTextWindow('Results', 10, 30, RESULTS_WINDOW_W,
                                         RESULTS_WINDOW_H)
    metrics_window = gui.ImguiTextWindow(
        'Metrics', 10, results_window.y + RESULTS_WINDOW_H + 10,
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
            coords, distance = camera.get_coords_of_object_xyxy(bbox)
            # Prediction of the color of the object
            quarter_width = (bbox[2] - bbox[0]) // 4
            quarter_height = (bbox[3] - bbox[1]) // 4
            inner_image = BoundingBox((bbox[0] + quarter_width,
                                       bbox[1] + quarter_height,
                                       bbox[2] - quarter_width,
                                       bbox[3] - quarter_height))

            test_histogram = color_recognition.color_histogram_of_image(imanip.extract_area_from_image(color_frame, inner_image))
            color_prediction = color_classifier.predict(test_histogram)
            # Format results for display
            if distance is None or coords is None:
                results_str.append(
                    f"{results.get_class_name(i)} ({results.get_conf(i):.2f}):\n\tnot in range\n\tcolor: {color_prediction}\n"
                )
            else:
                results_str.append(
                    f"{results.get_class_name(i)} ({results.get_conf(i):.2f}):\n\t{distance:.3f}mm\n\tcolor: {color_prediction}\n\t({coords[0]:.1f},{coords[1]:.1f},{coords[2]:.1f})"
                )
            
            edges = ed.edge_detection_rectangle_on_frame(color_frame, bbox, canny_low, canny_high)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            results_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = edges
                        
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

        window.draw_imgui_text_window(results_window)
        window.draw_imgui_text_window(metrics_window)

        window.end_drawing()

    print('Terminating...')
    window.terminate()
    camera.terminate()


if __name__ == '__main__':
    main()
