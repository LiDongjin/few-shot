from utils import dataloader
from tqdm import tqdm
from scipy.spatial import distance
import itertools
from sklearn.metrics import pairwise_distances_argmin, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def test_model(
    support_sentence_to_label,
    support_label_to_sentences,
    query_sentence_to_label,
    test_sentence_to_encoding,
    K=1,
):

    def get_sentence_to_embedding(sentence_to_encoding, sentence_to_label):
        sentence_embedding = []
        sentence_label = []
        for sentence, label in sentence_to_label.items():
            embedding = sentence_to_encoding[sentence]
            sentence_embedding.append(embedding)
            sentence_label.append(label)
        return sentence_embedding, sentence_label

    support_sentence_to_embedding, support_label = get_sentence_to_embedding(test_sentence_to_encoding, support_sentence_to_label)
    query_sentence_to_embedding, query_label = get_sentence_to_embedding(test_sentence_to_encoding, query_sentence_to_label)
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(support_sentence_to_embedding, support_label)
    y_pred = knn.predict(query_sentence_to_embedding)
    acc = accuracy_score(query_label, y_pred)

    return acc

def train_eval_model(
    cfg
):

    #load data
    support_sentence_to_label, support_label_to_sentences, query_sentence_to_label, test_sentence_to_encoding = dataloader.load_test_data(cfg)

    val_acc = test_model(
        support_sentence_to_label,
        support_label_to_sentences,
        query_sentence_to_label,
        test_sentence_to_encoding,
        K=1,
    )

    print(f"val_acc={val_acc:.4f}")

if __name__ == '__main__':
    from utils import common, configuration

    cfg_json_list = [
        "config/knn/1_shot.json",
        # "config/knn/5_shot.json",
        # "config/knn/10_shot.json",
        # "config/knn/15_shot.json",
        # "config/knn/20_shot.json",
    ]

    for cfg_json in cfg_json_list:
        cfg = configuration.triplet_config.from_json(cfg_json)
        print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        train_eval_model(cfg)

