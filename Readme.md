# Preparation 
1. Download the titanic dataset in your folder from https://www.kaggle.com/competitions/titanic/data 
2. Download and install anaconda from https://www.anaconda.com/
3. Create conda environment titanic as `conda create -n titanic python=3.11`
4. Change environment to titanic `conda activate titanic`
5. Install pybuilder `pip install pybuilder`
6. Refer to the pybuilder https://pybuilder.io/ and documents with tutorial from https://pybuilder.io/documentation/tutorial
7. Install mypy `pip install mypy`

# Build project from scratch
1. Create titanic folder and  change to the directory
2. Execute following command
3. `pyb --start-project`
	`pyb publish`
4. Open vscode in the given folder
5. install mypy types `mypy --install-types`

# Execution
Install dependencies
`pyb install_dependencies`

Execute the following command to build and install titanic library
`pyb install_dependencies && pyb && pyb publish && pip uninstall titanic && pip install target/dist/titanic-0.1.0/dist/titanic-0.1.0-py3-none-any.whl`

Execution of the code
`titanic_caller preprocess --train-data src/data/train.csv --test-data src/data/test.csv --output src/data/preprocessed_data.csv`
`titanic_caller train --preprocessed-data src/data/preprocessed_data.csv --output  src/data/titanic_models.pkl`
`titanic_caller evaluate --preprocessed-data src/data/preprocessed_data.csv --model  src/data/titanic_models.pkl --test-result src/data/gender_submission.csv`

To remove the package
`pip uninstall titanic`
