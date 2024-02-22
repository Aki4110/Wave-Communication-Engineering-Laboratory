# Ubuntu 20.04 LTSをベースイメージとする
FROM ubuntu:20.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive

# 基本的なパッケージのインストール
# 基本的なパッケージのインストールとクリーンアップ
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    python3-pip \
    lsb-release \
    sudo \
    software-properties-common \
    vim && \
    rm -rf /var/lib/apt/lists/*

# Python ライブラリのインストール
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Pythonのシンボリックリンクを設定
RUN ln -s /usr/bin/python3 /usr/bin/python

# ROSのインストール (Noeticを例とする)
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update && apt-get install -y ros-noetic-desktop-full

# ROS Tutorialsのインストール
RUN apt-get update && apt-get install -y ros-noetic-ros-tutorials

# ROSの環境設定
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"

# 作業ディレクトリの設定
WORKDIR /app
RUN mkdir -p /app/workspace

# コンテナ起動時に実行するコマンド
CMD ["/bin/bash"]
