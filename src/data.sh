#!/usr/bin/bash

wget \
-c https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv \
-O notebooks/2022-hw/data/car-price-data.csv
wget \
-c https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv \
-O notebooks/2022-hw/data/housing.csv
wget \
-c https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv \
-O notebooks/2023-hw/data/housing.csv
wget \
-c https://raw.githubusercontent.com/alexeygrigorev/datasets/master/AER_credit_card_data.csv \
-O notebooks/2022-hw/data/AER_credit_card_data.csv
wget \
-c https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/course-zoomcamp/cohorts/2022/05-deployment/homework/model1.bin \
-O src/2022-hw/05-deployment/model1.bin
wget \
-c https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/course-zoomcamp/cohorts/2022/05-deployment/homework/dv.bin \
-O src/2022-hw/05-deployment/dv.bin
wget \
https://github.com/alexeygrigorev/dino-or-dragon/releases/download/data/dino-dragon.zip \
-O notebooks/2022-hw/data/dino-dragon.zip
unzip -d notebooks/2022-hw/data/dino-dragon notebooks/2022-hw/data/dino-dragon.zip
kaggle datasets download \
thorgodofthunder/tvradionewspaperadvertising -f Advertising.csv \
-p src/linear-models/data/
