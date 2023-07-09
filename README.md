# deep-sort-master

This project offers an advanced learning system that sorts numbers using a seq2seq model with LSTM and a modified attention mechanism. The model concepts are based on Pointer Networks by Vinyals et al.

## Getting Started

To get started, first install all prerequisites by running `pip install -r requirements.txt` .

Next, generate the necessary data using [`generate.py`](generate.py).

## Training

Train your models by setting the appropriate parameters in [experiments/train.py](https://github.com/iFlyTech/deep-sort-master/blob/master/experiments/train.py). Call the train.run() method within [main.py](https://github.com/iFlyTech/deep-sort-master/blob/master/main.py), and finally execute `python main.py`.

## Evaluation

Once the model is trained, generate a test set using [generate.py](https://github.com/iFlyTech/deep-sort-master/blob/master/generate.py) and run [experiments/evaluate.py](https://github.