FOO=BAR
PREFIX_WK5=https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/course-zoomcamp/cohorts/2022/05-deployment/homework

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
	-c ${PREFIX_WK5}/model1.bin \
	-O src/2022-hw/05-deployment/model1.bin
	wget \
	-c ${PREFIX_WK5}/dv.bin \
	-O src/2022-hw/05-deployment/dv.bin
	wget \
	https://github.com/alexeygrigorev/dino-or-dragon/releases/download/data/dino-dragon.zip \
	-O notebooks/2022-hw/data/dino-dragon.zip
	unzip -d notebooks/2022-hw/data/dino-dragon notebooks/2022-hw/data/dino-dragon.zip

test: quality_checks
	pytest

quality_checks:
	isort src/
	black src/
	pylint --recursive=y src/

staging: test
	git add -u
	git status
