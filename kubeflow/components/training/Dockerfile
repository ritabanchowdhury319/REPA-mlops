FROM python
RUN python3 -m pip install pandas
RUN python3 -m pip install sklearn
COPY ./src/component-train.py /components/training/src/component-train.py
COPY xtrain.csv /components/training/xtrain.csv
COPY xtest.csv /components/training/xtest.csv
COPY selected_features.csv /components/training/selected_features.csv
RUN python3 /components/training/src/component-train.py
