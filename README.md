This paper introduces an algorithm for the automatic simulation of Bayesian dialogues [1], implemented in Python. The algorithm allows for the generation of dialogues that embody Bayesian reasoning. In addition to detailing the algorithm’s implementation and key features, we explore the probability of the emergence of non-trivial Bayesian dialogues by conducting random simulations on a vast sample of dialogues. The evaluation of the algorithm provides insights into its performance and the effectiveness of its simulations, underscoring its capability to encapsulate the core of Bayesian reasoning.

This work was presented at the Paris Workshop on Games, Decisions, and Language ([link](https://game-theory.u-paris2.fr/WS2023-program.html)) by Arsen Pidburachynskyi.

The file structure is organized as follows:
```
.
├── README.md
└── 
    └── ...
└── 
    └── ...
└── 
    └── ...
...

For running, the code, please install the neeeded dependencies with 

```shell
pip install -r requirements.txt
```

Set up your configuration in the `config.yaml` file, e.g.,

```
N: 13
A: [1,2,8,9,10,11]
w: 2
partition_1: [[1,2,3,4,5,6], [7,8,9,10,11], [12],[13]]
partition_2: [[1,2,7,8], [3,4,9,10], [5,6,11], [12,13]]
```

And run with 

```python
python run.py
```

Excpected results [with explainations]

[Optional, figure]

[1] We Can't disgree forever (pages 192--200). John D. Geanakoplos, Journal of Economic theory.