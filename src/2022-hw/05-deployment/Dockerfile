FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv
# WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY "q6_app.py" ./

EXPOSE 3998
CMD [ "waitress-serve", "--host=0.0.0.0", "--port=3998", "q6_app:app" ]
