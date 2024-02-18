# ベースイメージの指定
FROM ubuntu:22.04

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python ライブラリのインストール
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# アプリケーションのコピー
# COPY . /app
WORKDIR /app

# data ディレクトリの作成
RUN mkdir -p /app/workspace
RUN mkdir -p /app/Databox

# コマンド実行 (適切なコマンドに置き換えてください)
# CMD ["python3", "your_script.py"]
