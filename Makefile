all: test
format:
	black . -l 79
test:
	pytest .
