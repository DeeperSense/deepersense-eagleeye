# FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.13-py3
# FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
# FROM dustynv/ros:noetic-pytorch-l4t-r32.7.1



# ARM (l4t)
# FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.11-py3

# AMD64
# FROM rwthika/ros-torch:noetic-desktop-full-torch1.11.0-py-v23.06b
# FROM rwthika/ros-torch:noetic-perception-torch1.11.0-py
FROM rwthika/ros-torch:noetic-perception-torch1.11.0-py

# FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
# RUN apt-get update && apt-get install -y python3-tk
# Update locale
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# export LC_ALL=C.UTF-8
# export LANG=C.UTF-8

# Install libcanberra-gtk-module
# RUN apt-get update && apt-get install -y libcanberra-gtk-module libcanberra-gtk3-module


# # https://forums.developer.nvidia.com/t/invalid-public-key-for-cuda-apt-repository/212901/10
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 42D5A192B819C5DA
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# # RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN python3 -m pip install opencv-python



# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y python3-opencv









RUN pip3 install pillow --upgrade --force-reinstall --no-cache-dir
RUN pip3 install timm
RUN pip3 install matplotlib
RUN pip3 install scipy
RUN pip3 install ptflops
RUN pip3 install pandas
RUN pip3 install scapy






# YOLO / Segment Anyithing - FastSAM



# RUN pip3 install opencv-python


# ROS
RUN apt-get update
RUN apt-get install -y curl
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F42ED6FBAB17C654
RUN sudo apt update
RUN apt install -y ros-noetic-ros-base
# RUN pip3 install cvbridge3

# YOLO / Segment Anyithing - FastSAM



# RUN pip3 install opencv-python


# # ROS
# RUN apt-get update
# RUN apt-get install -y curl
# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F42ED6FBAB17C654
# RUN sudo apt update
# RUN apt install -y ros-noetic-ros-base
# # RUN pip3 install cvbridge3

# RUN mkdir -p /home/catkin_ws/src
# WORKDIR /home/catkin_ws/src
# RUN git clone --branch noetic https://github.com/ros-perception/vision_opencv.git




# RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
