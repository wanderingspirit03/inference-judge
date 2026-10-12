# The image behind targets/cpu-mock.yaml, and the ruler image run.sh measures
# from (IJ_RULER_IMAGE, default inference-judge-ruler:latest): python plus what
# bench/mock_server.py and the measuring side import. No bench code is baked
# in; run.sh mounts it read-only at run time, so the ruler hash covers it.
#   docker build -t inference-judge-mock:latest -t inference-judge-ruler:latest -f targets/cpu-mock.Dockerfile targets
FROM python:3.12-slim
RUN pip install --no-cache-dir --retries 10 --timeout 60 "httpx>=0.27" "numpy>=1.26" "pyyaml>=6.0" \
        "fastapi>=0.110" "uvicorn>=0.29" "tokenizers>=0.23"
