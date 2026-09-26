FROM python:3.11-slim

RUN pip install --no-cache-dir pandas scikit-learn xgboost

COPY train.py /usr/local/bin/train
RUN chmod +x /usr/local/bin/train 

ENV PATH="/usr/local/bin:${PATH}"
