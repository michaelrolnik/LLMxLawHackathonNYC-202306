FROM python:3.10

RUN mkdir /app
RUN mkdir /env
WORKDIR /app
ENV PYTHONPATH=${PYTHONPATH}:${PWD}

COPY . /app

ENV VIRTUAL_ENV=/env
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip3 install --upgrade pip && \
    pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry lock --no-update && \
    poetry install --no-interaction --no-ansi && \
    poetry run pip install jupyterlab

CMD ["poetry", "run", "jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--allow-root"]