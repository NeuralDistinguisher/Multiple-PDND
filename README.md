# Multiple-PDND
# Improved Polytopic Differential Neural Distinguishers for SIMON, SIMECK and SPECK Block Ciphers

This repository contains the implementation of Multiple Input Polytopic Differential Neural Distinguishers (pDND) for SIMON, SIMECK, and SPECK block ciphers. The provided code includes scripts for data generation, model training, and evaluation of the distinguishers.

## Directory Structure

- **Simeck32/**
  - `Models/`: Contains the trained models for SIMECK32.
  - `simeck.py`: Python script for data generation and training for SIMECK32.

- **Simon32/**
  - `Models/`: Contains the trained models for SIMON32.
  - `simon.py`: Python script for data generation and training for SIMON32.

- **Speck32/**
  - `Models/`: Contains the trained models for SPECK32.
  - `speck.py`: Python script for data generation and training for SPECK32.

## Prerequisites

The following Python libraries are required to run the scripts:

- numpy
- tensorflow
- pandas

You can install the required libraries using the following command:

```bash
pip install numpy tensorflow pandas
```

## Usage
To generate data and train the models for the block ciphers, follow these steps:

Clone the Repository:

```bash
git clone https://github.com/iman-mzm/Multiple-PDND.git
cd Multiple-PDND
```
Navigate to the Cipher Directory:

Depending on the block cipher you want to work with, navigate to the corresponding directory. For example, for SIMECK32:

```bash
cd Simeck32
```

Run Data Generation and Training Script:

Run the data generation and training script for the selected block cipher. For example, for SIMECK32:

```bash
python simeck.py
```
This will generate the required training data and start training the neural distinguisher.

## Model Evaluation:

To evaluate the models (or run real polytope differences experiment), comment out all sections related to network definition and training in the relevant function of the cipher algorithm analysis code. Then, execute the eval.py script in the directory of the desired cipher. For example, to evaluate the SIMECK32 model:

```bash
cd Simeck32
python eval.py
```
## Contribution
Feel free to fork the repository, make improvements, and submit pull requests. If you encounter any issues or have any questions, please open an issue on the GitHub repository.
