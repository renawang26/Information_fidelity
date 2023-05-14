# Information Fidelity

This repository is intended to provide the utility of fidelity assesment in interpreting and run inference.

## Setup

In a conda env, run:
```
conda env create -f ./environment.yml

conda activate Information_fidelity
```

## Inference
* Please update your OpenAI API key in similarity.py line 155 before running Python script
* Ensure Interpreter_Evaluation.xlsx is located in the same folder with similarity.py
* The similarity calculation result will be predocued and saved in similarity.xlsx

```
python similarity.py
```

