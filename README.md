# Object Coordinates Detection

## Goal

The goal of this project is to detect an object with characteristics specified by user (type and color) from the output of an Intel® Realsense™ D4XX depth camera, 
and to gather its coordinates in a 3D space relative to the camera.

## Requirememts

To use this library, you need to have:

- An Intel® Realsense™ D4XX depth camera
- The [Intel® Realsense™ SDK 2.0](https://www.intelrealsense.com/sdk-2/) installed on your computer
- Python 3.10 installed, along with the required Python libraries

To install the required libraries (except for the gui part), run the following command:

```shell
pip install numpy opencv-python torch pyrealsense2 ultralytics
```

## Demo usage

### Realtime detection with a GUI (gui-demo.py)

To install the required libraries, run the following command:

```shell
pip install glfw PyOpenGL PyOpenGL_accelerate
```

To run the demo, run the following command:

```shell
python3 gui-demo.py
```

### Realtime detection of object API usage (api-usage-demo.py)

To run the demo, run the following command:

```shell
python3 api-usage-demo.py
```

## Credits

Color recognition module is based on the work of Ahmet Özlü in 2018: [ahmetozlu/color_recognition](https://github.com/ahmetozlu/color_recognition)
