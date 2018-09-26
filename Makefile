.PHONY: clean data lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = face-recognition
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

RAW_DATA = data/raw/face_data_test.csv  data/raw/face_data_train.csv  data/raw/labels_test.csv  data/raw/labels_train.csv
FEATURE_DATA = data/processed/features_train.csv data/processed/features_test.csv
PCA_N_COEFFICIENTS = 40 
MODELS = models/knn_model.pkl
PREDICTIONS = predictions/prediction_knn_model.csv
KNN_N_COEFFICIENT = 1
#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	conda install --file requirements.txt

## Summarizes all models
summary: summary.csv

## Calculate predictions
predictions: $(PREDICTIONS)

## Train models
models: $(MODELS)

## Extract features
features: $(FEATURE_DATA) 

## Make Dataset
data: $(RAW_DATA)

## Delete all compiled Python files, Data and models
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find data -name "*" -type f -delete
	find models -name "*.pkl" -type f -delete
	find predictions -name "*.csv" -type f -delete
	rm summary.csv

## Lint using flake8
lint:
	flake8 src

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

###################################
#### data rules
###################################
data/raw/face_data_test.csv: src/data/make_dataset.py
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw

data/raw/face_data_train.csv: src/data/make_dataset.py
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw

data/raw/labels_train.csv: src/data/make_dataset.py
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw

data/raw/labels_test.csv: src/data/make_dataset.py
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw

###################################
#### features rules
###################################

data/processed/features_train.csv: src/features/build_features.py $(RAW_DATA)
	$(PYTHON_INTERPRETER) $< data/raw/face_data_train.csv data/raw/face_data_test.csv $(FEATURE_DATA) $(PCA_N_COEFFICIENTS) 

data/processed/features_test.csv: src/features/build_features.py $(RAW_DATA)
	$(PYTHON_INTERPRETER) $< data/raw/face_data_train.csv data/raw/face_data_test.csv $(FEATURE_DATA) $(PCA_N_COEFFICIENTS)

#### model rules
models/knn_model.pkl: $(FEATURE_DATA) src/models/knn_model.py
	$(PYTHON_INTERPRETER) src/models/knn_model.py data/processed/features_train.csv data/raw/labels_train.csv $@ $(KNN_N_COEFFICIENT)

#### prediction rules
predictions/prediction_knn_model.csv: models/knn_model.pkl src/models/predict_model.py
	$(PYTHON_INTERPRETER) src/models/predict_model.py $< data/processed/features_test.csv $@

#### summary rules
summary.csv: $(PREDICTIONS) src/models/summarize_models.py
	$(PYTHON_INTERPRETER) src/models/summarize_models.py predictions/ data/raw/labels_test.csv $@

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
