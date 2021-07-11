from utils import dataloader
from tqdm import tqdm
from scipy.spatial import distance
import itertools
from sklearn.metrics import pairwise_distances_argmin, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def eval_model(
    train_sentence_to_label,
    train_label_to_sentences,
    train_sentence_to_encoding,
    test_sentence_to_label,
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

    train_sentence_to_embedding, train_label = get_sentence_to_embedding(train_sentence_to_encoding, train_sentence_to_label)
    test_sentence_to_embedding, test_label = get_sentence_to_embedding(test_sentence_to_encoding, test_sentence_to_label)
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(train_sentence_to_embedding, train_label)
    y_pred = knn.predict(test_sentence_to_embedding)
    acc = accuracy_score(test_label, y_pred)
    # min_index = pairwise_distances_argmin(test_sentence_to_embedding, train_sentence_to_embedding, metric='cosine')
    #
    # predicted_label = [train_label[x] for x in min_index]
    # assert len(predicted_label) == len(test_label)
    # num_correct = 0
    # for i in range(len(test_label)):
    #     if predicted_label[i] == test_label[i]:
    #         num_correct += 1
    # acc = num_correct / len(test_label)
    return acc

def train_eval_model(
    cfg
):

    #load data
    train_sentence_to_label, train_label_to_sentences, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding = dataloader.load_data(cfg)

    val_acc = eval_model(
        train_sentence_to_label, 
        train_label_to_sentences, 
        train_sentence_to_encoding, 
        test_sentence_to_label, 
        test_sentence_to_encoding,
        K=5,
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

