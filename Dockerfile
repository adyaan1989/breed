FROM docker:19.03

FROM python:3.7
WORKDIR /usr/src/app
COPY requirements.txt /usr/src/app/
RUN pip install -r requirements.txt
COPY . /usr/src/app