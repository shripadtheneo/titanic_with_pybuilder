# Preparation 
1. Download the titanic dataset in your folder from https://www.kaggle.com/competitions/titanic/data 
2. Download and install anaconda from https://www.anaconda.com/
3. Create conda environment titanic as `conda create -n titanic python=3.11`
4. Change environment to titanic `conda activate titanic`
5. Install pybuilder `pip install pybuilder`
6. Refer to the pybuilder https://pybuilder.io/ and documents with tutorial from https://pybuilder.io/documentation/tutorial

# Build project from scratch
1. Create titanic folder and  change to the directory
2. Execute following command
3. `pyb --start-project`
	`pyb publish`
4. Open vscode in the given folder

# Execution
Build test
`pyb`

Execute the following command to build and install titanic library
`pyb && pyb publish && pip install target/dist/titanic-0.1.0/dist/titanic-0.1.0-py3-none-any.whl`

Execution of the code
`titanic_caller --data-dir /home/shripad/source_code/titanic/src/data --test-data test.csv --train-data train.csv --test-result gender_submission.csv`

To remove the package
`pip uninstall titanic`