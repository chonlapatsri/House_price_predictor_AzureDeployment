#I specify the parent base image which is the python version 3.10.12
FROM python:3.10.12-slim

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
    && apt-get -y install gcc make \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --no-cache-dir --upgrade pip

# set work directory
WORKDIR /app

# copy requirements.txt
COPY ./requirement.txt /app/requirement.txt

# copy html files
COPY ./app/templates/home.html /app/templates/home.html
COPY ./app/templates/result.html /app/templates/result.html

# copy css file
COPY ./app/static/style.css /app/static/style.css

# copy app executable
COPY ./app/app_v2.py /app/app_v2.py

# Install project requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt

# Copy custom code file and trained model in pickle format
COPY ./ML_models/custom_class_predictor.py /app/custom_class_predictor.py
COPY ./ML_models/rndf_house_price_estimator.pkl /app/rndf_house_price_estimator.pkl

# set app port
EXPOSE 8080

ENTRYPOINT [ "python" ] 

# Run app.py when the container launches
CMD [ "app_v2.py","run","--host","0.0.0.0"] 