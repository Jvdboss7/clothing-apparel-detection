FROM continuumio/miniconda3

RUN apt update -y && apt install awscli -y

WORKDIR /clothing
COPY . /clothing

RUN apt-get update && pip install --upgrade pip
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
RUN conda install -c conda-forge pycocotools
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

#RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
RUN pip install -e .

CMD ["python","app.py"]
