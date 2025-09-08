# GPU runtime base (update tag as needed)
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

ARG USERNAME=ubuntu

ARG DEBIAN_FRONTEND=noninteractive

# ---- Base OS & tooling -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    build-essential pkg-config libssl-dev \
    git openssh-client ca-certificates curl vim \
  && rm -rf /var/lib/apt/lists/*


# ---- Install uv (system-wide) -----------------------------------------------
# The install script honors UV_INSTALL_DIR.
ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Helpful defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Make uv more tolerant of slow networks
ENV UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=8 \
    UV_CONCURRENT_DOWNLOADS=4

# ---- Install Rust (system-wide via rustup) -----------------------------------
# Install to /usr/local so both root and 'ubuntu' can use it.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.89.0 \
 && chmod -R a+rX /usr/local/rustup /usr/local/cargo

USER $USERNAME

ENV PATH="/usr/local/cargo/bin:${PATH}"

# Per-user caches + writable build target
ENV CARGO_HOME=/home/$USERNAME/.cargo
ENV CARGO_TARGET_DIR=/home/$USERNAME/.cargo-target
RUN mkdir -p $CARGO_HOME $CARGO_TARGET_DIR

# Add a persistent uv cache dir for $USERNAME
ENV UV_CACHE_DIR=/home/$USERNAME/.cache/uv
RUN mkdir -p "$UV_CACHE_DIR"

# ---- Project files -----------------------------------------------------------
WORKDIR /workspace
RUN chown -R $USERNAME:$USERNAME /workspace

# 1) Copy
COPY --chown=$USERNAME:$USERNAME . /workspace/

# Build the Rust workspace (release build unless you need debug)
RUN cargo build --release
ENV MAYBENOT=$CARGO_TARGET_DIR/release/maybenot

# 3) Create the project venv *inside def-ml* and install deps
WORKDIR /workspace/def-ml
ENV PATH="/workspace/def-ml/.venv/bin:${PATH}"
ENV PATH="${PATH}:${CARGO_TARGET_DIR}/release/"
RUN --mount=type=cache,target=$UV_CACHE_DIR,uid=1000,gid=1000,mode=0775

RUN uv sync

# Issues w. permissions...
USER root


