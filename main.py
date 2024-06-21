from camera import DepthCamera
from model import YoloModel
import gui
import gpu_utils
import glfw

WIDTH, HEIGHT = 640, 480
FPS = 30

RESULTS_WINDOW_W, RESULTS_WINDOW_H = 180, 200
METRICS_WINDOW_W, METRICS_WINDOW_H = 180, 100
SLIDERS_WINDOW_W, SLIDERS_WINDOW_H = 180, 80

def main():
    camera = DepthCamera(width=WIDTH, height=HEIGHT, fps=30)

    model = YoloModel('./bluecups.pt')

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

    while not window.should_close():
        camera.update_frame()
        color_frame = camera.get_color_frame()
        
        results = model.predict_frame(color_frame, iou=iou_thres, conf=conf_thres)
        
        results_str = ''
        print(results.results_count())
        for i in range(results.results_count()):
            coords, distance = camera.get_coords_of_object_xyxy(results.get_box_coords(i))
            if distance is None or coords is None:
                results_str = results_str.join(f"{results.get_class_name(i)} ({results.get_conf(i):.2f}): not in range\n")
            else:
                results_str = results_str.join(f"{results.get_class_name(i)} ({results.get_conf(i):.2f}):\n\t{distance:.3f}mm\n\t({coords[0]:.1f},{coords[1]:.1f},{coords[2]:.1f})\n")
        results_str = results_str.lstrip()
        results_window.set_text(results_str)

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
