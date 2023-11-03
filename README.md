## CPDT
The source code of paper, "CPDT: A Novel Cluster-based Paired Decision Tree for Identifying Biomedical Entity Interactions".

## Requirements

```bash
Python==3.7.9
sklearn==1.0.2
numpy==1.19.5
scipy==1.5.2
```

## Usage

The preprocessing of the dataset and the training of CPDT are encapsulated in a file named main.py, which you can run directly to get the experimental results.

```bash
python main.py
```

The Results are stored in the results directory, with a csv file for each dataset that stores the results of the 5-fold cross-validation.
