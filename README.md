# Object Coordinates Detection

## Goal

The goal of this project is to detect an object with characteristics specified by user (type and color) from the output of an Intel® Realsense™ D4XX depth camera, 
and to gather its coordinates in a 3D space relative to the camera.

## Requirememts

To use this library, you need to have:

- An Intel® Realsense™ D4XX depth camera
- The [Intel® Realsense™ SDK 2.0](https://www.intelrealsense.com/sdk-2/) installed on your computer
- Python 3.10 installed, along with the required Python libraries

## Installation

You can install this library using the following command (assuming you are in the root directory):

```shell
cd .. && pip install ./object-detection
```

You can then use this package with the module named 'aifinder', with each file in aifinder being a submodule.

## Demo usage

### Realtime detection with a GUI (gui-demo.py)

To run the demo, run the following command:

```shell
python3 gui-demo.py
```

It will open a window, showing the object detection in realtime, with each bounding box having an edge detection demonstration.
You can change the prediction parameters (intersection-over-union and minimal confidence) in realtime.

### Realtime detection of object API usage (api-usage-demo.py)

To run the demo, run the following command:

```shell
python3 api-usage-demo.py
```

It shows a very basic usage of the api, where you can add API calls in the main loop.

## Credits

Color recognition module is based on the work of Ahmet Özlü in 2018: [ahmetozlu/color_recognition](https://github.com/ahmetozlu/color_recognition)
