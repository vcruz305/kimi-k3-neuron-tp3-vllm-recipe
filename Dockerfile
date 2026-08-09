ARG CUDA_BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92
FROM ${CUDA_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=8

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VLLM_TARGET_DEVICE=cuda \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=4

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential ca-certificates curl git libnuma-dev python3.12 \
       python3.12-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/k3-venv
ENV PATH=/opt/k3-venv/bin:${PATH}

WORKDIR /opt/recipe
COPY . /opt/recipe

RUN python3 scripts/check_bundle.py \
    && scripts/prepare_sources.sh /opt/k3-sources \
    && scripts/build_from_source.sh /opt/k3-sources \
    && python3 scripts/assert_runtime.py --allow-no-gpu

EXPOSE 8008
CMD ["/bin/bash"]
