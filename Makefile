install:
	pip install --upgrade pip && pip install -r requirements.txt

lint:
	pylint --disable=R,C *.py

test:
	# --cov=my_function test_*.py for details on a function
	# test_*.py → python tests files are prefixed as is
	#  --disable-warnings, if needed
	# -vvv for verbose
	# -s for disabling capturing
	#     (allows print statements to be shown in the console even if the test passes)
	python -m pytest -vvv -s

format:
	# --force-exclude '<FILE_OR_FOLDER>' if needed (env, imported, models...)
	black *.py *.ipynb

all: install lint test format

# personal and not click-setup-installed commands
# api_launch:
# 	python $(CURDIR)/blueprint_fast_api.py

# mlflow_run:
# 	python mlflow_folder/my_mlf_script.py && mlflow ui