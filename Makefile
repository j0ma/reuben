coverage-report:
	py.test --verbose -s --flakes --vulture --cov-report html:coverage-report --cov . reuben/test.py
	firefox coverage-report/index.html 

look-at-coverage-report:
	firefox coverage-report/index.html 

test:
	py.test -s  --flakes --vulture --verbose reuben/test.py
