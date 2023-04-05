FROM tensorflow/tensorflow:latest-gpu
ARG model
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["./train_model.sh"]

