from flask import Flask, Response, make_response, render_template, request
from aifinder.depth_finder import DepthFinder
import cv2
import threading
import time

print('restart')

app = Flask(__name__)

WIDTH, HEIGHT = 1280, 720
FPS = 30

depth_finder = DepthFinder(WIDTH, HEIGHT, 30, "yolov8s.pt")
quit = False

def update() -> None:
    while not quit:
        if depth_finder is None:
            return
        depth_finder.update()

update_thread = threading.Thread(target=update)

def convert_frame_to_jpeg(frame):
    return cv2.imencode('.jpeg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 50])[1]

def get_frame(fps: int):
    color_frame = None
    jpeg_frame = None
    string_frame = None
    refresh_rate = 1/fps
    while True:
        color_frame = depth_finder.frame
        if color_frame is None:
            return '\r\n'
        jpeg_frame = convert_frame_to_jpeg(color_frame)
        string_frame = jpeg_frame.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+string_frame+b'\r\n')
        time.sleep(refresh_rate)

@app.route('/live')
def live():
    fps = max(1, min(request.args.get('fps', default=FPS, type=int), FPS))
    return render_template('live.html', fps=fps)

@app.route('/live-stream')
def live_stream():
    fps = max(1, min(request.args.get('fps', default=FPS, type=int), FPS))
    return Response(get_frame(fps), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/frame')
def color_frame():
    response = make_response(convert_frame_to_jpeg(color_frame))
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers['Content-Type'] = 'image/jpeg'
    return response

@app.route('/objects')
def all_objects():
    return depth_finder.to_object_list()

@app.route('/objects/<string:class_name>')
def single_object(class_name: str):
    conf = request.args.get('conf', default=0.0, type=float)
    results = depth_finder.find_object_by_name(class_name, min_conf=conf)
    if results is None:
        return {}
    return {
        'x': results[0],
        'y': results[1],
        'z': results[2],
    }

if __name__ == '__main__':
    update_thread.start()
    app.run(threaded=True)
    quit = True
    depth_finder.terminate()
