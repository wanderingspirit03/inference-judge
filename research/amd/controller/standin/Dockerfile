# The stand-in GPU machine: a fresh Ubuntu 24.04 with sshd and no GPU.  Its
# entry point applies the cloud-init user-data the driver sends (the host
# key and root's authorized key), as cloud-init does on a droplet, and starts
# sshd with that host key only.  It runs with NET_ADMIN (fake_do.py) so that
# setup-box.sh applies the real firewall rule for cand (controller/box/ij-fw)
# in the container's own network namespace.
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      openssh-server python3 python3-yaml rsync sudo git ca-certificates procps iptables \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd && rm -f /etc/ssh/ssh_host_*
COPY standin-init.py /standin-init.py
CMD ["python3", "/standin-init.py"]
