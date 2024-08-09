from aifinder.camera import DepthCamera
from aifinder import gui
import glfw
import json
from threading import Thread
import time
from ur_api import *

# Characteristics of the depth camera
WIDTH, HEIGHT = 1280, 720
FPS = 30
FRAME_DURATION = 1/FPS

results = dict()
currently_moving = False
POINTS = 4
current_point = 0
finished = False
AXIS = ['o', 'x', 'y', 'z']

ARM_CAL_POS = [ (0.6, 0.0, 0.1), (0.7, 0.0, 0.1), (0.6, 0.1, 0.1), (0.6, 0.0, 0.2) ]
RX, RY, RZ = (0.0, 3.0, 0.0)


camera = DepthCamera(width=WIDTH, height=HEIGHT, fps=FPS)
window = gui.Window("Calibration", WIDTH, HEIGHT)

def mouse_button_callback(window, button: int, action: int, mods: int):
    global current_point, currently_moving, finished
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and not currently_moving and current_point < POINTS:
        x, y = glfw.get_cursor_pos(window)
        x, y = int(x), int(y)
        coords, _ = camera.get_coords_of_pixel(x, y)
        if coords is None:
            return

        print(coords)
        
        results[AXIS[current_point]] = { 'x': coords[0], 'y': coords[1], 'z': coords[2] }
        current_point += 1
        currently_moving = True
        if current_point == POINTS:
            finished = True

glfw.set_mouse_button_callback(window.window, mouse_button_callback)

def move_arm():
    global currently_moving
    init_arm()
    currently_moving = True
    for i in range(POINTS):        
        x, y, z = ARM_CAL_POS[i]
        # move(x, y, z, RX, RY, RZ)
        print(f"Moving to {x}, {y}, {z}")
        time.sleep(4)
        print(f"Moved to {x}, {y}, {z}")
        currently_moving = False
        while not currently_moving:
            time.sleep(FRAME_DURATION)
    ox, oy, oz = ARM_CAL_POS[0]
    move(ox, oy, oz)
    close_arm()

robot_thread = Thread(target=move_arm)
robot_thread.start()

while not window.should_close() and not finished:
    camera.update_frame()

    frame = camera.get_color_frame_as_ndarray()
    if frame is None:
        continue

    window.begin_drawing()

    window.draw_background_from_mem(frame, WIDTH, HEIGHT)

    window.end_drawing()

with open('calibration.json', 'w+') as calibration_file:
    calibration_file.write(json.dumps(results))

robot_thread.join()

window.terminate()
camera.terminate()
