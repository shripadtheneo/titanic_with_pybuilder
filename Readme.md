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

# Update Needed
- HOME_DIR Path has some bug when calling as command from command line so need to check that part
- Some more update needed for the purpose of accepting user arguments
- Test code needed to be written
- pip install to be added at the time of pyb