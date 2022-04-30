# The implementation for distributed DGNN algorithms

## Create the conda environment
```
conda create -n 'DGNN' python=3.8
```

## Install the requirements
```
pip install -r requirements.txt
```

## Run test for local training
### For training DGNN on a single node:
```
python test.py --json-path=./parameters.json --test-type=local 
```

### For training DGNN on multiple nodes:
```
python test.py --json-path=./parameters.json --test-type=dp 
```
