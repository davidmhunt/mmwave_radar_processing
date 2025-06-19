# mmwave_radar_processing
python libraries for processing raw mmWave radar data

## Installation
In order for the code to work properly, the following steps are required
1. Install correct version of python
2. Install mmWaveRadarProcessing using Poetry

### 1. Setup Python environment

#### Deadsnakes PPA (requires sudo access)
1. On ubuntu systems, start by adding the deadsnakes PPA to add the required version of python.
```
sudo add-apt-repository ppa:deadsnakes/ppa
```

2. Update the package list
```
sudo apt update
```

3. Install python 3.10 along with the required development dependencies
```
sudo apt install python3.10 python3.10-dev
```

The following resources may be helpful [Deadsnakes PPA description](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa), [Tutorial on Deadsnakes on Ubuntu](https://preocts.github.io/python/20221230-deadsnakes/)

#### Conda (Backup)
1. If conda isn't already installed, follow the [Conda Install Instructions](https://conda.io/projects/conda/en/stable/user-guide/install/index.html) to install conda
2. Use the following command to download the conda installation (for linux)
```
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
```
3. Run the conda installation script (-b for auto accepting the license)
```
bash Anaconda3-2023.09-0-Linux-x86_64.sh -b
```
3. Once conda is installed, create a new conda environment with the correct version of python
```
conda create -n mmWaveRadarProcessing python=3.10
```

### 2. Clone mmwave_radar_processing
```
git clone https://github.com/davidmhunt/mmwave_radar_processing.git
```

Initialize the submodule
```
cd mmwave_radar_processing
git submodule update --init
```
### 3. Install mmwave_radar_processing using Poetry

#### Installing Poetry:
 
1. Check to see if Python Poetry is installed. If the below command is successful, poetry is installed move on to setting up the conda environment

```
    #for now ensure that it is lower than version 2.0 (TOML files are out of date)
    poetry --version
```
2. If Python Poetry is not installed, follow the [Poetry Install Instructions](https://python-poetry.org/docs/#installing-with-the-official-installer). On linux, Poetry can be installed using the following command:
```
curl -sSL https://install.python-poetry.org | python3 - --version 1.8.4
```

If you are using poetry over an ssh connection or get an error in the following steps, try running the following command first and then continuing with the remainder fo the installation.
```
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

#### Installing mmwave_radar_processing
Navigate to the mmwave_radar_processing foler (this folder) and execute the following command

```
poetry install --extras "submodules"
```

If you get an an error saying: "Failed to unlock the collection!", execute the following command in the terminal:
```
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

#### Updating mmwave_radar_processing
If the pyproject.toml file is updated, the poetry installation must also be updated. Use the following commands to update the version of poetry
```
poetry lock --no-update
poetry install
```

### Using .env for Project Directories

In order to use any datasets in your computer's directory, you must first create a .env file and mark where the dataset files can be found.

1. Create a .env file in your project's root directory. This will file will not be uploaded to GitHub when you commit your changes.
2. Inside the .env file, add these variables
```
DATASET_DIRECTORY=/path/to/datasets
CONFIG_DIRECTORY=/path/to/mmwave_radar_processing/configs
MOVIE_TEMP_DIRECTORY=/path/to/movie/temp/movie_temp_directory
ANALYZER_TEMP_DIRECTORY=/path/to/temp/analyzer_directory
```
3. Replace the example text with the path to your directory