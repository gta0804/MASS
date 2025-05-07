sources = src
format:
	isort $(sources) tests
	black $(sources) tests

lint:
	flake8 $(sources) tests
	mypy $(sources) tests

clean:
	rm -rf .mypy_cache .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox build dist site
	rm -rf coverage.xml junit-report.xml .coverage
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

export-requirements:
	pdm export --without-hashes -o requirements.txt --prod
	pdm export --without-hashes --no-default -o requirements-dev.txt --dev
