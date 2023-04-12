FROM tensorflow/tensorflow:1.15.0-gpu-py3
WORKDIR /ids
COPY . .
RUN pip install -r requirements.txt
RUN mkdir Data
#ENTRYPOINT ["bash train_model.sh"]

