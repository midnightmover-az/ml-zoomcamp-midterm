FROM python:3.8.12-slim

RUN pip install pandas
RUN pip install "scikit-learn==1.3.1"
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

#COPY logisticRegressionModel.bin /'logisticRegressionModel.bin'
COPY xgBoostModel.bin /xgBoostModel.bin

RUN pipenv install --system --deploy

COPY ["predict.py", "", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
