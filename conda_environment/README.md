<h2><b> How to Recreate the Conda Environment Used in this Project </b></h2>
Author: Tamer Abousoud<br><br>


There are many dependencies required for the code in this repository (Tensorflow, TF-Agents, PyBullet, and much more). To make it easier to configure a local conda environment on your local system, a copy of the conda environment used for this project is included. The `robotics_conda_env.yml` file in this directory is a YAML file containing the pip and conda packages I am currently using. To recreate the environment on your system, download the file locally, make sure you have conda installed and enter the following command in a terminal:

```
conda env create --file robotics_conda_env.yml
```
(*Note: The full path to `robotics_conda_env.yml` should be prepended if it is not in the current working directory.*)

<br><h3><u>Additional Notes</u></h3>

- This environment was only run on my local Linux desktop machine, it has not been tested on other systems.
- The environment in the `.yml` file is provided "as-is" and may contain unnecessary packages that were used or considered at one point.
- The conda environment I'm using is constantly changing, so the YAML in this folder may be a bit outdated. You may get an `ImportError` or `ModuleNotFoundError` when trying to run code in this repo and need to manually install some packages. I will try to update the YAML (or add new ones) whenever there are significant changes or updates to the environment. 

<br><h3><u>Setup guide for Mac</u></h3>
Author: Tam Nguyen<br><br>

Conda environment does not work properly on Mac BigSur version. To get the conda setup, following the following steps:

1. Test conda on your terminal. If it returns an error `conda command not found`, moving to step 2. Otherwise, just need to change the directory in the yml file run the command create env above.
```
conda list
```
2. Make sure you already download your Anaconda or miniconda on your machine. If not, download [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)

- On your terminal,run the 1st command to get your user name
- Then, replace `yourUserrName` in the 2nd command with your actual user name 
```
echo $USER
export PATH="/Users/yourUserrName/anaconda3/bin:$PATH"
```

- Do some checkings to make sure conda works properly
```
conda --version
conda env list
```
- Once done, you can navigate to PyBullet_Test folder and try to run some python file there. If it throws any error related to `package could not found`, just need to do either of the following command.

```
conda install <packageName>
pip3 install <packageName>
```

