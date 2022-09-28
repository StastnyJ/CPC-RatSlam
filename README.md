# INSTALLATION



## Prerequisites

- ROS (this repository was developed and tested on Ubuntu 20.04.4 LTS x86_64 with ROS **noetic**) - [Installation guide](http://wiki.ros.org/noetic/Installation/Ubuntu)
- Python3 (code was tested with Python 3.8.10) - [Installation guide](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-programming-environment-on-an-ubuntu-20-04-server)
- OpenCV - [Installation guide](https://vitux.com/opencv_ubuntu/)
- Pytorch - [Installation guide](https://www.linode.com/docs/guides/pytorch-installation-ubuntu-2004/)
- Irrlicht - [Download link](https://sourceforge.net/projects/irrlicht/files/Irrlicht%20SDK/)

## Required python packages
- numpy - [PyPI](https://pypi.org/project/numpy/)
- scipy - [PyPI](https://pypi.org/project/scipy/)
- sklearn - [PyPI](https://pypi.org/project/scikit-learn/)
- statistics - [PyPI](https://pypi.org/project/statistics/)
- torch - [PyPI](https://pypi.org/project/torch/)
- matplotlib - [PyPI](https://pypi.org/project/matplotlib/)
- cv2 - [PyPI](https://pypi.org/project/opencv-python/)
- cv_bridge - [PyPI](https://pypi.org/project/cvbridge3/)
- colormath - [PyPI](https://pypi.org/project/colormath/)
- tf - [PyPI](https://pypi.org/project/tf/)
- plotly.express - [PyPI](https://pypi.org/project/plotly-express/)

## Installation guide

1. Install all prerequisites and required python packages
2. create a catkin workspace
3. clone this repository in the src folder in the created workspace
4. clone [OpenRatSLAM](https://github.com/davidmball/ratslam) repository ito the src folder next to this repository
5. run `catkin_make` command

# EXPERIMENT REPLICATION 