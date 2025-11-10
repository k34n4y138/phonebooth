FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
LABEL org.opencontainers.image.authors="Zakaria Moumen <keanay@1337.ma>"
RUN apt update && apt install -y curl gnupg


RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-cuda-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" | tee /etc/apt/sources.list.d/cuda.list
RUN apt update && apt install -y libcudnn9-cuda-12


WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY phonebooth.py demo.html /app/

ENV COQUI_TOS_AGREED=1
RUN python phonebooth.py #load models

CMD ["uvicorn", "phonebooth:app", "--host", "0.0.0.0", "--port", "8000"]

