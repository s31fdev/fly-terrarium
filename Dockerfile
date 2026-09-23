# Python 3.12 on Debian trixie (same distro as the official nelyepfl/flygym image,
# whose virtualenv actually runs Python 3.14 - see README).
FROM python:3.12-slim-trixie@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9

# Mesa libraries for headless software rendering (no GPU needed).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libegl1 libgl1 libgl1-mesa-dri libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --root-user-action=ignore -r /tmp/requirements.txt

ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PYTHONUNBUFFERED=1

WORKDIR /app
CMD ["python", "run_route.py"]
