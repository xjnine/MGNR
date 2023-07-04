# MNGR
A Multi-Granularity Neighbor Relationship and Its Application in KNN Classification and Clustering Methods
## MGKNN
Improve the KNN method by combining the granular-ball model and multi-granularity neighbor relationship to achieve adaptive estimation of the $k$ value.
### Files
These program mainly containing:
  - a  real dataset folder named "datasets".
  - a python file named "main.py" is the main function file.
  - other python files
### Usage
  Run the main.py to test the code on the dataset and evaluate the classification results using ACC scores.
## NARD
Improve the density-based clustering methods by combining the granular ball model and multi-granularity neighbor relationship to achieve adaptive density estimation for each sample.
### Files
These program mainly containing:
  - proposed algorithms folder named "ProposedAlgorithm" containing $NARD$ and $SDGS$.
  - all modified algorithms folder named "ModifiedAlgorithm" containing $DensityPeak$, $DBSCAN$, $DADC$ and $HCDC$.
  - a python file named "run.py" in folder named "ModifiedAlgorithm" is the main function file.
  - other python files
### Usage
Run the run.py to test the code on the synthetic dataset or real dataset and evaluate the clustering results.
## Requirements
### Installation requirements (Python 3.9)
  - Pycharm 
  - Linux operating system or Windows operating system
  - sklearn, numpy, pandas
## Dataset Format
The dataset being tested should be given in a csv file of the following format:
- For real dataset, in every line, there are d+1 numbers: the label of the data point and the d coordinate values, where the label is an integer from 0 to n-1.
- For synthetic dataset, in every line, there are d+1 numbers:  the d coordinate values and the label of the data point, where the label is an integer from 0 to n-1.
