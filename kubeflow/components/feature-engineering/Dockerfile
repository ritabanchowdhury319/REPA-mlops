FROM python
RUN python3 -m pip install pandas
RUN python3 -m pip install sklearn
COPY ./src/component.py /components/feature-engineering/src/component.py
COPY xtrain.csv /components/feature-engineering/xtrain.csv
COPY xtest.csv /components/feature-engineering/xtest.csv
RUN python3 /components/feature-engineering/src/component.py
