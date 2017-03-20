.PHONY: init

init:
	pip install -r requirements.txt

test:
	# This runs all of the tests. To run an individual test, run py.test with
	# the -k flag, like "py.test -k test_path_is_not_double_encoded"
	# To disable capturing stdout, use -s flag
	py.test tests

test-watch:
	# watch environment and re-test after changes are made
	py.test tests -n 4 --looponfail
