# hash:sha256:73ddf74a50d38f70f7942f215a0e24a44c3b7900bc28f4325e00717779cb00c0
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.36.0 \
    elementpath==4.4.0 \
    npc-session==0.1.34 \
    spikeinterface[full]==0.100.6 \
    wavpack-numcodecs==0.1.5


COPY postInstall /
RUN /postInstall
