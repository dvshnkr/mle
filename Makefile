# Some of these work on Windows...
# LOCAL_DATE = %DATE:~6,4%-%DATE:~3,2%-%DATE:~0,2%
# LOCAL_TIME = %TIME: =0%
# TIME_FMT = %$(LOCAL_TIME):~0,2%-%$(LOCAL_TIME):~3,2%
# DATE_TIME = $(LOCAL_DATE)-$(LOCAL_TIME)

# This works on bash (git bash on Windows, too)
# LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
# LOCAL_IMAGE_NAME=stream-model-duration:${LOCAL_TAG}

FOO=BAR
PREFIX=https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/course-zoomcamp/cohorts/2022/05-deployment/homework

all:
	echo ${FOO}
	echo all

dotenv:
# 	creates an environment file by appending nothing to a non-existant file
#	functionally similar to how touch .env can be used in unix
	type nul >> ".env"

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
	wget \
	-c ${PREFIX}/model1.bin \
	-O src/2022-hw/05-deployment/model1.bin
	wget \
	-c ${PREFIX}/dv.bin \
	-O src/2022-hw/05-deployment/dv.bin

test: quality_checks
	pytest

build: test
	echo build

run: build
	echo run

quality_checks:
	isort src/
	black src/
	pylint --recursive=y src/

staging: test
	git add -u
	git status
