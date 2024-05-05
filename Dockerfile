FROM python:3.10-slim
WORKDIR /nmt
COPY . /nmt/
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD [ "gradio", "app.py" ]