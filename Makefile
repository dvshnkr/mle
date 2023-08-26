# Some of these work on Windows...
# LOCAL_DATE = %DATE:~6,4%-%DATE:~3,2%-%DATE:~0,2%
# LOCAL_TIME = %TIME: =0%
# TIME_FMT = %$(LOCAL_TIME):~0,2%-%$(LOCAL_TIME):~3,2%
# DATE_TIME = $(LOCAL_DATE)-$(LOCAL_TIME)

# This works on bash (git bash on Windows, too)
# LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
# LOCAL_IMAGE_NAME=stream-model-duration:${LOCAL_TAG}

all: data
	echo all

data:
	wget \
	-c https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv \
	-O notebooks/2022-hw/data/car-price-data.csv
	wget \
	-c https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv \
	-O notebooks/2022-hw/data/housing.csv
	wget \
	-c https://raw.githubusercontent.com/alexeygrigorev/datasets/master/AER_credit_card_data.csv \
	-O notebooks/2022-hw/data/AER_credit_card_data.csv

test:
	pytest src/tests/

build: test
	echo build

run: build
	echo run

quality_checks:
	isort src/
	black src/
	pylint --recursive=y src/
