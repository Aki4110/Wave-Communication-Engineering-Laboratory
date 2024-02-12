# ベースイメージの指定
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# vi エディタのインストール
RUN apt-get update && apt-get install -y vim

# Python ライブラリのインストール
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# アプリケーションのコピー
COPY . /app
WORKDIR /app

# コマンド実行
#CMD ["python3", "your_script.py"]
