from camera import DepthCamera
from model import YoloModel
import gui
import gpu_utils
import glfw
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import numpy as np
import image_manipulation as imanip

WIDTH, HEIGHT = 640, 480
FPS = 30

RESULTS_WINDOW_W, RESULTS_WINDOW_H = 180, 200
METRICS_WINDOW_W, METRICS_WINDOW_H = 180, 80
SLIDERS_WINDOW_W, SLIDERS_WINDOW_H = 180, 80

def main():
    camera = DepthCamera(width=WIDTH, height=HEIGHT, fps=30)

    model = YoloModel('yolov8s.pt')

    window = gui.Window("Yolov8", WIDTH, HEIGHT)

    results_window = gui.ImguiTextWindow('Results', 10, 30, RESULTS_WINDOW_W, RESULTS_WINDOW_H)
    metrics_window = gui.ImguiTextWindow('Metrics', 10, results_window.y + RESULTS_WINDOW_H + 10, METRICS_WINDOW_W, METRICS_WINDOW_H)

    t0 = glfw.get_time()
    t = 0
    fps = window.get_fps()
    gpu_total_mem = gpu_utils.query_gpu_total_mem()
    gpu_used_mem = gpu_utils.query_gpu_used_mem(units=False)
    gpu_util = gpu_utils.query_gpu_utilization()
    metrics_window.set_text(f"FPS: {fps}\nGPU Usage: {gpu_utils.query_gpu_utilization()}%\nGPU mem: {gpu_utils.query_gpu_used_mem(units=False)}/{gpu_total_mem}")

    iou_thres = 0.5
    conf_thres = 0.8
    is_sliders_expand = True
    is_sliders_close = True
    
    results_str = []
    color_prediction = ''

    while not window.should_close():
        camera.update_frame()
        color_frame = camera.get_color_frame_as_ndarray()

        if color_frame is None:
            continue


        results = model.predict_frame(color_frame, iou=iou_thres, conf=conf_thres)

        results_str = []
        for i in range(results.results_count()):
            bb_box = results.get_box_coords(i)
            coords, distance = camera.get_coords_of_object_xyxy(bb_box)
            color_histogram_feature_extraction.color_histogram_of_test_image(imanip.extract_area_from_image(color_frame, bb_box[0], bb_box[1], bb_box[2], bb_box[3]))
            color_prediction = knn_classifier.main("training.data", "test.data")
            if distance is None or coords is None:
                results_str.append(f"{results.get_class_name(i)} ({results.get_conf(i):.2f}):\n\tnot in range\n\tcolor: {color_prediction}\n")
            else:
                results_str.append(f"{results.get_class_name(i)} ({results.get_conf(i):.2f}):\n\t{distance:.3f}mm\n\tcolor: {color_prediction}\n\t({coords[0]:.1f},{coords[1]:.1f},{coords[2]:.1f})")
        results_window.set_text('\n'.join(results_str))

        t = glfw.get_time()
        fps = window.get_fps()
        if t - t0 > 1.0:
            t0 = t
            gpu_util = gpu_utils.query_gpu_utilization()
            gpu_used_mem = gpu_utils.query_gpu_used_mem(units=False)
        metrics_window.set_text(f"FPS: {fps:.1f}\nGPU Usage: {gpu_util}%\nGPU mem: {gpu_used_mem}/{gpu_total_mem}")

        window.begin_drawing()

        window.draw_background_from_mem(results.render(), WIDTH, HEIGHT)

        gui.im.set_next_window_position(10, metrics_window.y + METRICS_WINDOW_H + 10, condition=gui.im.ONCE)
        gui.im.set_next_window_size(SLIDERS_WINDOW_W, SLIDERS_WINDOW_H, condition=gui.im.ONCE)
        if is_sliders_close:
            is_sliders_expand, is_sliders_close = gui.im.begin("Prediction parameters")
            if is_sliders_expand:
                _, iou_thres = gui.im.slider_float(
                    "iou", iou_thres,
                    min_value=0.0, max_value=1.0,
                    format="%.2f"
                )
                _, conf_thres = gui.im.slider_float(
                    "conf", conf_thres,
                    min_value=0.0, max_value=1.0,
                    format="%.2f"
                )
            gui.im.end()

        window.draw_imgui_text_window(results_window)
        window.draw_imgui_text_window(metrics_window)

        window.end_drawing()

    print('Terminating...')
    window.terminate()
    camera.terminate()

if __name__ == '__main__':
    main()
