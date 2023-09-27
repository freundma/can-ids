FROM tensorflow/tensorflow:latest-gpu
WORKDIR /ids
COPY . .
RUN pip install -r requirements.txt
RUN mkdir Data
#ENTRYPOINT ["bash train_model.sh"]

