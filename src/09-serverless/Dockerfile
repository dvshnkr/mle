# FROM public.ecr.aws/lambda/python:3.9
FROM python:3.9-slim

RUN python -m pip install --upgrade pip
RUN python -m pip install tflite-runtime

COPY "requirements.txt" .

RUN python -m pip install -r requirements.txt

COPY ["dino_dragon_10_0.899.tflite", "serve.py", "./" ]

COPY ["app.py", "lambda_function.py", "./"]

EXPOSE 8990

# CMD [ "lambda_function.lambda_handler" ]
CMD [ "waitress-serve", "--host=0.0.0.0", "--port=8990", "app:app" ]
