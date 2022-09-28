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
4. clone [OpenRatSLAM](https://github.com/davidmball/ratslam) repository into the src folder next to this repository
5. run `catkin_make` command
6. run `source devel/setup.sh`

# EXPERIMENT REPLICATION

## Classic experiment

This experiment measures the total accuracy of the approach, time and memory performance and average false positive error. The PR curves and final paths from RatSLAM integration are not included.

1. *if not first use:* clear results from previous experiments
2. `roslaunch colored_point_cloud_rat_slam_ros firstStageOnly.launch` for 1st stage only or `roslaunch colored_point_cloud_rat_slam_ros bothStages.launch` for both stages
3. `rosbag play <bag file name>` (rosbag files are uploaded to the "TODO path")
4. *wait for the rosbag file to finish*
5. Results can be found in:
   - **Total accuracy**: Total accuracy is printed directly to the console
   - **Information about all false positives:** in the folder `./anal/FPs/` are images of all generated false positives. Detailed information about false positives is generated in the file `./anal/FPs/fpDetails.txt` 
   - **Time performance:** time performance is generated in files `./anal/buildingTimes.txt` and `./anal/matchingTimes.txt`
   - **Memory consumption:** can be found in the file `./anal/memory.txt`
  

### **Data visualization**

- To visualize data, run script `./scripts/tools/graphMaker.py`.
## PR Curve generation


## RatSLAM integration