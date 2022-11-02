FROM debian:latest
WORKDIR "/root"
COPY requirements.txt "/root/"
RUN apt update && \
    apt install -y python3 python3-pip wget libgl1 libglib2.0-0 --no-install-recommends && \
    pip install -r requirements.txt && \
    mkdir -p output
COPY AnimeGAN* run.py "/root/"
COPY input/* "/root/input/"
CMD ["python3", "run.py"]

# apt update
# apt install python3 python3-pip wget --no-install-recommends
# pip install opencv-python numpy onnxruntime
# cd root
# wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VPAPI84qaPUCHKHJLHiMK7BP_JE66xNe' -O AnimeGAN_Hayao.onnx
# wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=17XRNQgQoUAnu6SM5VgBuhqSBO4UAVNI1' -O AnimeGANv2_Hayao.onnx
# wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=10rQfe4obW0dkNtsQuWg-szC4diBzYFXK' -O AnimeGANv2_Shinkai.onnx
# wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1X3Glf69Ter_n2Tj6p81VpGKx7U4Dq-tI' -O AnimeGANv2_Paprika.onnx

# aws ecr-public get-login-password --region us-east-1 | podman login --username AWS --password-stdin public.ecr.aws/o3i8v6z7
# podman build -t cse291_virtualization_animegan .
# podman tag cse291_virtualization_animegan:latest public.ecr.aws/o3i8v6z7/cse291_virtualization_animegan:latest && podman push public.ecr.aws/o3i8v6z7/cse291_virtualization_animegan:latest

# podman run -it --rm -v ./output:/root/output cse291_virtualization_animegan
