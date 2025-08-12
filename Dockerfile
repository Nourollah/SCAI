FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install pixi dependencies
RUN apt-get update && \
    apt-get install -y git curl bzip2 && \
    rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"
RUN pixi self-update

WORKDIR /app

# Copy all project files
COPY ./ ./

# Install dependencies and project
RUN pixi install
RUN pixi run post_install

CMD ["pixi", "run", "train", "configs/Experiment-1/schnet.yml"]