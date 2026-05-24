from glob import glob
from pathlib import Path

from setuptools import find_packages, setup


package_name = "rage_cage_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=["scripts"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/model", [str(path) for path in Path("model").glob("*.onnx")]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="orin",
    maintainer_email="orin@example.com",
    description="Ping-pong ball 3D perception from ball 2D detection and table ArUco pose.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "perception_node = scripts.perception_node:main",
        ],
    },
)
