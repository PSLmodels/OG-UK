all: test
format:
	black . -l 79
test:
	pytest .
install:
	pip3 install -e .
	pip3 install policyengine-uk
