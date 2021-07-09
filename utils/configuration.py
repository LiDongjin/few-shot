import json
from typing import NamedTuple


class triplet_config(NamedTuple):

    exp_id: str = None
    seed_num: int = 0

    #dataset params
    train_path: str = None
    test_path: str = None

    # model params
    max_length: int = 64
    encoding_size: int = 768 #bert embedding size
    embedding_size: int = 30 #output embedding size
    triplet_margin: float = 0.4
    learning_rate: float = 3e-5

    # training params
    num_classes: int = None
    samples_per_class: int = None
    val_subset: int = 10000  # 从测试集随机选取多少个样本进行验证
    total_updates: int = 10000
    eval_interval: int = 100
    num_epochs: int = 100
    minibatch_size: int = 20


    @classmethod
    def from_json(cls, file_path):
        return cls(**json.load(open(file_path, 'r')))


class mlp_config(NamedTuple):
    exp_id: str = None
    seed_num: int = 0

    #dataset params
    train_path: str = None
    test_path: str = None

    # model params
    max_length: int = 64
    encoding_size: int = 768 # bert embedding size
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    decay_gamma: float = 0.95
    model: str = None # "LR"

    # training params
    num_classes: int = None
    samples_per_class: int = None
    val_subset: int = 10000
    num_epochs: int = 100
    eval_interval: int = 100
    minibatch_size: int = 20


    @classmethod
    def from_json(cls, file_path):
        return cls(**json.load(open(file_path, 'r')))
