# Information_fidelity

This repository is intended to provide the interpretation fidelity utility and run inference.

## Setup

In a conda env , run:
```
conda env create -f ./environment.yml

conda activate Information_fidelity
```

## Inference
* Please update your OpenAI API key in similarity.py line 155 before run Python script
* Check is Interpreter_Evaluation.xlsx locate in same folder with similarity.py
* The similarity calculation result will put in similarity.xlsx

```
python similarity.py
```

