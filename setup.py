from setuptools import setup 

setup(
    name="aifinder",
    version="1.0.0",
    packages=[
        "aifinder"
    ],
    package_dir={
        "aifinder": "aifinder"
    },
    install_requires=[
        "numpy",
        "pyrealsense2",
        "opencv-python",
        "PyOpenGL",
        "imgui",
        "glfw",
        "torch",
        "ultralytics"
    ],
    python_requires="==3.10.*"
)
