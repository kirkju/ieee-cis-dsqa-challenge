# A bare Linux machine, for tools/container_check.py.
#
# Nothing of the evaluation's environment is in the image. Python, the pinned
# packages and the models are put on a volume by tools/pod_bootstrap.sh, run
# in a container of this image as it is run on a rented pod. The entry is then
# run from that volume with the network cut and the disk read-only.
#
# The evaluation's own image is not public. What is known of it is that it
# lacks the system library cv2 needs (libGL), and so does this one.
FROM ubuntu:22.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*
