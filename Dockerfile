FROM --platform=linux/amd64 my_huggingface_atten:v1 AS surgvu25Cat2

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PATH="/home/user/.local/bin:$PATH"

RUN echo "deb https://mirrors.aliyun.com/ubuntu noble main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/ubuntu noble-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/ubuntu noble-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/ubuntu noble-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    rm -rf /etc/apt/sources.list.d/* && \
    apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version && \
    python3 -m pip --version


RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

USER root
RUN set -ex && \
    apt-get update && \
    apt-get install -y --no-install-recommends proxychains4 && \
    sed -i '/^socks/ s/^/#/' /etc/proxychains4.conf && \
    echo "socks5 10.70.21.10 12306" >> /etc/proxychains4.conf && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER user
RUN proxychains4 python3 -m pip install \
    --no-cache-dir \
    --no-color \
    --user \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user model /opt/app/model
COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python3", "inference.py"]
