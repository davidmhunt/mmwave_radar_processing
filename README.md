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

3. Install python 3.12 via deadsnakes
```
sudo apt install python3.12 python3.12-dev
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
    poetry --version
```
2. If Python Poetry is not installed, follow the [Poetry Install Instructions](https://python-poetry.org/docs/#installing-with-the-official-installer). On linux, Poetry can be installed using the following command:
```
curl -sSL https://install.python-poetry.org | python3 -
```

If you are using poetry over an ssh connection or get an error in the following steps, try running the following command first and then continuing with the remainder fo the installation.
```
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

Finally, make sure that poetry is installing everything into the environment (i.e. not using system packages)
```
poetry config virtualenvs.options.system-site-packages false
```

#### Installing mmwave_radar_processing
```
cd mmwave_radar_processing
poetry install --with submodules
```

#### Updating mmwave_radar_processing
If the pyproject.toml file is updated, the poetry installation must also be updated. Use the following commands to update the version of poetry
```
poetry lock
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

## Running Tests

To run the unit tests, use the following command:

```bash
poetry run pytest tests/
```

Note: Since the tests involve GUI components, if you are running in a headless environment (e.g., SSH without X11 forwarding), you may need to set the `QT_QPA_PLATFORM` environment variable:

```bash
QT_QPA_PLATFORM=offscreen poetry run pytest tests/
```

## Launching the GUI
To launch the GUI, use the `launch_mmwave_viewer.py` script:
```bash
poetry run python scripts/launch_mmwave_viewer.py   
```

### Configuration
The GUI uses configuration files located in the `gui_configs` directory:
- `dataset_params.yaml`: Defines the default dataset to load and its associated radar configuration.
- `processor_params.yaml`: Defines parameters for the various signal processing modules.

### Dataset Paths
By default, the GUI looks for datasets relative to the project root directory. You can specify a different dataset using the `--dataset-path` argument or by updating `dataset_params.yaml`. Absolute paths are also supported.

### Command Line Arguments
- `--dataset-params`: Path to dataset parameters YAML (default: `gui_configs/dataset_params.yaml`)
- `--processor-params`: Path to processor parameters YAML (default: `gui_configs/processor_params.yaml`)
- `--dataset-path`: Override the dataset path (default: from `dataset_params.yaml`)
- `--config-name`: Override the radar config file (default: from `dataset_params.yaml`)
- `--log-level`: Set logging level (default: INFO)

### More Documentation
For more detailed information on the GUI architecture, extending the viewer, and processor details, please refer to the documentation in the `docs/` folder:
- [GUI Documentation](docs/GUI.md)
- [Processors Documentation](docs/processors.md)

## Velocity Analysis

To perform velocity estimation analysis, use the `scripts/test_vel_estimation.py` script. This script runs the velocity estimation pipeline on a dataset and generates performance metrics and plots compared to ground truth.

### Usage

```bash
poetry run python scripts/test_vel_estimation.py --config-name velocity_analysis_config.yaml
```

**Arguments:**
- `--config-name`: Name of the configuration file located in `analyzer_configs/`. Default: `velocity_analysis_config.yaml`.
- `--plot-time-series-errors` / `--no-plot-time-series-errors`: Enable/disable plots showing X, Y, Z, and Norm errors over time. Default: Enabled.
- `--plot-distributions` / `--no-plot-distributions`: Enable/disable CDF and Histogram plots of the error distribution. Default: Enabled.
- `--plot-histograms` / `--no-plot-histograms`: Enable/disable explicit histogram plots for X, Y, and Z errors. Default: Enabled.
- `--plot-comparison` / `--no-plot-comparison`: Enable/disable plots comparing Estimated vs Ground Truth velocities for each axis. Default: Enabled.
- `--plot-stats` / `--no-plot-stats`: Enable/disable plots for R2 statistics and Inlier percentages over time. Default: Enabled.

### Configuration

The analysis is configured via `analyzer_configs/velocity_analysis_config.yaml`. Key sections include:

- **dataset**:
    - `path`: Root directory of the dataset.
    - `name`: Name of the specific dataset folder.
- **radar**: 
    - `config_file`: Radar config file name.
    - `array_geometry`: Antenna array geometry (e.g., "ods").
- **processors**: Parameter dictionaries for specific processors:
    - `velocity_estimator`: Thresholds for R2 and inliers.
    - `point_cloud_generator`: Detection and CFAR parameters.
- **analysis**:
    - `start_idx`: Start frame index for analysis.
    - `end_idx`: End frame index.
    - `error_method`: Method for error calculation, either "signed" (Estimated - GT) or "absolute" (|Estimated - GT|).
- **transformation**:
    - `uav_vel_matrix`: 3x3 matrix to transform *Estimated* velocities into the desired frame.
    - `gt_vel_matrix`: 3x3 matrix to transform *Ground Truth* velocities into the desired frame.

### Output

The script will:
1. Print summary statistics (Mean, Median, RMSE) for X, Y, Z, and Norm velocity errors to the console.
2. Display plots for:
   - Velocity Estimation vs Ground Truth (Time Series)
   - R2 Statistics and Inlier Percentage
   - Error Distribution (Histograms and CDFs)

## Multi-Dataset Velocity Analysis

To compute summary statistics and produce distribution plots across multiple datasets, use the `scripts/test_multi_vel_estimation.py` script. This script processes every dataset defined in the configuration over all their frames to provide a global view of the velocity estimator's accuracy.

### Usage

```bash
poetry run python scripts/test_multi_vel_estimation.py --config-name multi_dataset_velocity_analysis_config.yaml
```

**Arguments:**
- `--config-name`: Name of the configuration file located in `analyzer_configs/`. Default: `multi_dataset_velocity_analysis_config.yaml`.
- `--plot-distributions` / `--no-plot-distributions`: Enable/disable CDF and Histogram plots of the error distribution. Default: Enabled.
- `--plot-histograms` / `--no-plot-histograms`: Enable/disable explicit histogram plots for X, Y, and Z errors. Default: Enabled.
- `--plot-time-series-errors` / `--no-plot-time-series-errors`: Enable/disable plots showing X, Y, Z, and Norm errors over time. Default: Disabled.
- `--plot-comparison` / `--no-plot-comparison`: Enable/disable plots comparing Estimated vs Ground Truth velocities for each axis. Default: Disabled.
- `--plot-stats` / `--no-plot-stats`: Enable/disable plots for R2 statistics and Inlier percentages over time. Default: Disabled.

### Configuration

The analysis is configured via `analyzer_configs/multi_dataset_velocity_analysis_config.yaml`. Key differences from the single-dataset config include:

- **datasets**: A list of dataset dictionaries, each with a `path` and `name`, replacing the single `dataset` item.
- **analysis**: The `start_idx` and `end_idx` parameters are omitted, as the script automatically processes all frames in each provided dataset.

### Output

The script will:
1. Print summary statistics (Mean, Median, RMSE) for X, Y, Z, and Norm velocity errors across all datasets to the console.
2. Display plots for:
   - Error Distribution (Histograms and CDFs, if enabled)
   - Velocity Estimation vs Ground Truth (Time Series, if enabled)
   - R2 Statistics and Inlier Percentage (if enabled)