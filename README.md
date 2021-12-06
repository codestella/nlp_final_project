# SNU NLP Final Project
팀원: 김장현, 안형서, 양서연, 이욱재

### Requirements
- PyTorch 1.10.0
- transformers

### Notes
- ```/jupyter``` contains jupyper files for each task.
- We perform basic hyperparameter tuning based on ```main.py```:
```
python main.py --lr [float] --batch_size [int] --tune full --size large
```
- To train smaller models, set ```--size base```.
- To freeze backbone feature extractor, set ```--tune linear```.
- We collect experiment logs in ```/experiment_log```.
