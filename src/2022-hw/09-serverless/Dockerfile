FROM svizor42/zoomcamp-dino-dragon-lambda:v2

# RUN python -m pip install --upgrade pip
RUN pip install pillow
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl

COPY ["lambda_function.py", "./"]

ENV MODEL_NAME=dino-vs-dragon-v2.tflite

CMD [ "lambda_function.lambda_handler" ]
