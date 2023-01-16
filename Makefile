all: test
format:
	black . -l 79
test:
	pytest .
install:
	pip install -e .
	pip install policyengine-uk
