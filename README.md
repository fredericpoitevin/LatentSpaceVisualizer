# LatentSpaceVisualizer

The Latent Space Visualizer is an interactive Python notebook that visualizes the latent space of an image dataset. The dataset contains a collection of latent variables that describes a set of labeled images. 

Two variables of the latent space are displayed in a 2D scatter plot. When the mouse is positioned near a point in the scatter plot, the image corresponding to that point is displayed to the side.

<img src="figures/Figure1.png" />

## Getting started

Clone this repository.

```bash
git clone https://github.com/compSPI/LatentSpaceVisualizer.git
```

Change the current working directory to the root of this repository.

```bash
cd LatentSpaceVisualizer
```

Download from the Anaconda Cloud and install the Python environment that has the dependencies required to run the code.

```bash
conda env create compSPI/compSPI
```

Activate the environment.

```bash
conda activate compSPI
```

Install the kernel.

```bash
python -m ipykernel install --user --name compSPI --display-name "Python (compSPI)"
```

Exit the environment.

```bash
conda deactivate
```

## Running the notebook

Run jupyter notebook.

```bash
jupyter notebook 
```

Open the tutorial notebook ```latent_space_visualizer.ipynb```.

Change the Python kernel to ```compSPI```.

Set ```dataset_file``` to an HDF5 file containing the dataset.

```python
dataset_file = '../data/cspi_synthetic_dataset_diffraction_patterns_1024x1040.hdf5'
```

Run the notebook.

## Installation

The project package can be installed by running the following command.

```bash
python setup.py install
```

## Code structure

The relevant files and folders in this repository are described below:

- ```README.md```: Highlights the usefulness of the Latent Space Visualizer. 

- ```latent_space_visualizer.ipynb```:  Provides a tutorial notebook for using the Latent Space Visualizer.

- ```latent_space_visualizer/```: Contains the Python file required to run the notebook.
