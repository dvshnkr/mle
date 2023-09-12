FOO=BAR
PREFIX_WK5=

all:
	echo ${FOO}
	echo all

dotenv:
# 	creates an environment file by appending nothing to a non-existant file
#	functionally similar to how touch .env can be used in unix
	type nul >> ".env"

data:
	bash src/data.sh

test: quality_checks
	pytest

quality_checks:
	isort src/
	black src/
	pylint --recursive=y src/

staging: test
	git add -u
	git status
