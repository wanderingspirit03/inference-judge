# AMD's tuned engine (proaditya/ATOM qwen38-megakernels at 352229d6, flydsl 0.4.0.dev973 with the wheel's own
# frontend) as a submission.  The base image ij-atom-tuned:dev973 is built on the GPU machine by
# measurements/2026-11-17/gpu-session/build-tuned.sh; this file only makes it usable by a non-root uid, which
# is how controller/box/ij-docker runs every engine (the caller's uid, HOME=/tmp, no capabilities).
# The build has no network (ij-docker builds with --network none).
FROM ij-atom-tuned:dev973
# kernel caches, the megakernel cache, the drafter objects and aiter's JIT directory are written at run time
RUN for d in /opt/qwen38 /opt/atom /opt/aiter /work; do [ -e "$d" ] && chmod -R a+rwX "$d" || true; done
# ATOM's chat answer gives text only; with return_token_ids it now also gives the answer's ids (the megakernel
# path has no logprobs, and the judge reads ids)
COPY patch_token_ids.py /tmp/patch_token_ids.py
RUN python3 /tmp/patch_token_ids.py /opt/atom/atom/entrypoints/openai/serving_chat.py && rm /tmp/patch_token_ids.py
WORKDIR /opt/atom
