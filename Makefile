all: test
format:
	black . -l 79
test:
	pytest .
install:
	pip install policyengine-uk
	pip install --upgrade jsonschema[format-nongpl]
	pip install -e .[dev]
