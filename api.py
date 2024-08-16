#!/usr/bin/python3
import os

# Removes the Yolo terminal output
os.environ['YOLO_VERBOSE'] = 'False'

from flask import Flask, Response, make_response, render_template, request, url_for
from aifinder.depth_finder import DepthFinder, Point3D
import cv2
import threading
import time

app = Flask(__name__)

WIDTH, HEIGHT = 1280, 720
FPS = 30

depth_finder = DepthFinder(WIDTH, HEIGHT, 30, "./bluecups.pt")
quit = False

def update() -> None:
    while not quit:
        if depth_finder is None:
            return
        depth_finder.update()

update_thread = threading.Thread(target=update)

def convert_frame_to_jpeg(frame, color_space_conv = cv2.COLOR_RGB2BGR):
    return cv2.imencode('.jpeg', cv2.cvtColor(frame, color_space_conv), [cv2.IMWRITE_JPEG_QUALITY, 90])[1]

def get_frame(fps: int):
    color_frame = None
    jpeg_frame = None
    string_frame = None
    refresh_rate = 1/fps if fps > 0 else 1/FPS
    while True:
        color_frame = depth_finder.frame
        if color_frame is None:
            return '\r\n'
        jpeg_frame = convert_frame_to_jpeg(color_frame)
        string_frame = jpeg_frame.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+string_frame+b'\r\n')
        time.sleep(refresh_rate)

def get_prediction_frame(fps: int):
    prediction_frame = None
    jpeg_frame = None
    string_frame = None
    refresh_rate = 1/fps if fps > 0 else 1/FPS
    while True:
        prediction_frame = depth_finder.render_prediction()
        if prediction_frame is None:
            return '\r\n'
        jpeg_frame = convert_frame_to_jpeg(prediction_frame)
        string_frame = jpeg_frame.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+string_frame+b'\r\n')
        time.sleep(refresh_rate)

def get_depth_frame(fps: int):
    depth_frame = None
    jpeg_frame = None
    string_frame = None
    refresh_rate = 1/fps if fps > 0 else 1/FPS
    while True:
        depth_frame = depth_finder.render_depth()
        if depth_frame is None:
            return '\r\n'
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
        jpeg_frame = convert_frame_to_jpeg(depth_colormap)
        string_frame = jpeg_frame.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+string_frame+b'\r\n')
        time.sleep(refresh_rate)

@app.route('/live')
def live():
    fps = max(1, min(request.args.get('fps', default=FPS, type=int), FPS))
    return render_template('live.html', fps=fps, host=request.host, stream='live-stream')

@app.route('/live-stream')
def live_stream():
    fps = max(1, min(request.args.get('fps', default=FPS, type=int), FPS))
    return Response(get_frame(fps), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live-prediction')
def live_prediction():
    fps = max(1, min(request.args.get('fps', default=FPS, type=int), FPS))
    return render_template('live.html', fps=fps, host=request.host, stream='live-prediction-stream')

@app.route('/live-prediction-stream')
def live_prediction_stream():
    fps = max(1, min(request.args.get('fps', default=FPS, type=int), FPS))
    return Response(get_prediction_frame(fps), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live-depth')
def live_depth():
    fps = max(1, min(request.args.get('fps', default=FPS, type=int), FPS))
    return render_template('live.html', fps=fps, host=request.host, stream='live-depth-stream')

@app.route('/live-depth-stream')
def live_depth_stream():
    fps = max(1, min(request.args.get('fps', default=FPS, type=int), FPS))
    return Response(get_depth_frame(fps), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/frame')
def render_color_frame():
    color_frame = depth_finder.frame
    response = make_response(convert_frame_to_jpeg(color_frame).tobytes())
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers['Content-Type'] = 'image/jpeg'
    return response

@app.route('/frame-prediction')
def render_frame_prediction():
    prediction_frame = depth_finder.render_prediction()
    response = make_response(convert_frame_to_jpeg(prediction_frame).tobytes())
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers['Content-Type'] = 'image/jpeg'
    return response  

@app.route('/frame-depth')
def render_depth_frame():
    depth_frame = depth_finder.render_depth()
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
    response = make_response(convert_frame_to_jpeg(depth_colormap).tobytes())
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers['Content-Type'] = 'image/jpeg'
    return response  

@app.route('/frame-depth-raw')
def render_depth_frame_raw():
    return depth_finder.render_depth().tolist()

@app.route('/objects')
def all_objects():
    return depth_finder.to_object_list()

@app.route('/class_names')
def class_names():
    return depth_finder.get_classes_names()

@app.route('/objects/<string:class_name>')
def single_object(class_name: str):
    conf = request.args.get('conf', default=0.0, type=float)
    color = request.args.get('color', default=None, type=str)
    camera_space = request.args.get('camera_space', default='false', type=str)
    arm_space = False if camera_space == 'true' else True
    print(arm_space)
    results = None
    if color is None:
        results = depth_finder.find_object_by_name(class_name, arm_space, min_conf=conf)
    else:
        results = depth_finder.find_object_by_name_and_color(class_name, color, arm_space, min_conf=conf)
    if results is None:
        return {}
    return {
        'x': results.x,
        'y': results.y,
        'z': results.z,
    }

@app.route('/pixel-coords')
def pixel_coords():
    x = request.args.get('x', default=0, type=int)
    y = request.args.get('y', default=0, type=int)
    cam_coords, arm_coords = depth_finder.get_coords_of_pixel(x, y)
    return {
        'arm_coords': {
            'x': arm_coords.x,
            'y': arm_coords.y,
            'z': arm_coords.z,
        },
        'cam_coords': {
            'x': cam_coords.x,
            'y': cam_coords.y,
            'z': cam_coords.z,
        }
    }

@app.route('/convert-coords')
def convert_coords():
    x = request.args.get('x', default=0.0, type=float)
    y = request.args.get('y', default=0.0, type=float)
    z = request.args.get('z', default=0.0, type=float)
    result = depth_finder.converter.to_coords(Point3D(x,y,z))
    return {
        'x': result.x,
        'y': result.y,
        'z': result.z,
    }

def has_no_empty_params(rule):
    defaults = rule.defaults if rule.defaults is not None else ()
    arguments = rule.arguments if rule.arguments is not None else ()
    return len(defaults) >= len(arguments)


@app.route("/site-map")
def site_map():
    links = []
    for rule in app.url_map.iter_rules():
        # Filter out rules we can't navigate to in a browser
        # and rules that require parameters
        if "GET" in rule.methods and has_no_empty_params(rule):
            url = url_for(rule.endpoint, **(rule.defaults or {}))
            links.append((url, rule.endpoint))
    # links is now a list of url, endpoint tuples

    return links


if __name__ == '__main__':
    update_thread.start()
    app.run(host="0.0.0.0", threaded=True)
    quit = True
    depth_finder.terminate()
