
FROM python:3.12.5-slim


WORKDIR /app


COPY . /app


RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8888

ENV PYTHONUNBUFFERED=1

CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--allow-root"]
