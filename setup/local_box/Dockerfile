# The local box: a fresh Ubuntu 24.04 machine, as close to a rented accelerator
# box as a container on a laptop gets, minus the accelerator.  setup/bootstrap.sh
# runs inside it for a target with accelerator.kind none (targets/cpu-mock.yaml).
#
# Built and driven by setup/local_box/run_local_box.sh.  Run it --privileged:
# the box runs its own dockerd (Docker-in-Docker), because bench/run.sh
# bind-mounts paths that exist only inside the box.
#
# Everything here is what a real box image already has or what bootstrap would
# install from the network anyway (apt packages, uv); baking it in only makes
# repeated runs fast.  Nothing in here is the harness: bootstrap installs that.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8 LANG=C.UTF-8

# bootstrap's s_apt list, plus what a cloud image ships (sudo, python3-yaml),
# plus dockerd itself (the real box image has Docker preinstalled).
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
      build-essential ca-certificates cmake curl git gnupg jq ninja-build numactl pciutils \
      python3 python3-venv python3-pip python3-yaml python3-numpy python3-httpx \
      rsync tmux unzip wget sudo procps psmisc iproute2 iptables shellcheck \
      docker.io \
 && rm -rf /var/lib/apt/lists/*

# uv, pinned (bootstrap installs it with curl | sh when it is missing).
RUN pip3 install --no-cache-dir --break-system-packages uv==0.11.29

# A uv cache every user can share, so the bench venv does not re-download its
# wheels on every run.  sudo reads /etc/environment through pam_env, so the
# `sudo -u bench -H bash -lc` calls in bootstrap see it too.
ENV UV_CACHE_DIR=/var/cache/uv UV_LINK_MODE=copy
RUN mkdir -p /var/cache/uv && chmod 1777 /var/cache/uv \
 && printf 'UV_CACHE_DIR=/var/cache/uv\nUV_LINK_MODE=copy\n' >> /etc/environment \
 && uv venv -q --python /usr/bin/python3 /tmp/warm \
 && uv pip install -q --python /tmp/warm/bin/python \
      'httpx>=0.27' 'numpy>=1.26' 'pyyaml>=6.0' 'fastapi>=0.110' 'uvicorn>=0.29' 'huggingface_hub>=0.23' \
      'pytest>=8.0' 'pytest-asyncio>=0.23' setuptools \
 && rm -rf /tmp/warm && chmod -R a+rwX /var/cache/uv

# The box's init: dockerd in the background, then the container just stays up.
# run_local_box.sh drives every stage with `docker exec`.
COPY box_init.sh /usr/local/sbin/box_init.sh
RUN chmod 0755 /usr/local/sbin/box_init.sh
VOLUME /var/lib/docker
ENTRYPOINT ["/usr/local/sbin/box_init.sh"]
