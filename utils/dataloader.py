import itertools
import numpy as np
import random
import torch
from models.sub_layer import bert_encoding

def get_sentence_to_label(csv_file):
    lines = open(csv_file, 'r').readlines()
    sentence_to_label = {}
    label_to_sentences = {}
    
    for line in lines:
        parts = line.strip().split('\t')
        label = int(parts[1].split(',')[-1])
        sentence = parts[0]
        sentence_to_label[sentence] = label

        if label in label_to_sentences:
            label_to_sentences[label].append(sentence)
        else:
            label_to_sentences[label] = [sentence]

    return sentence_to_label, label_to_sentences


def get_train_subset(train_label_to_sentences, samples_per_class, seed_num):
    random.seed(seed_num)
    return {train_label: random.sample(sentences, samples_per_class) for train_label, sentences in train_label_to_sentences.items()}

# 测试集的子集用于测试
def get_test_subset(test_sentence_to_label, val_subset, seed_num):
    random.seed(seed_num)
    if val_subset < len(test_sentence_to_label):
        return {test_sentence: label for test_sentence, label in random.sample(test_sentence_to_label.items(), val_subset)}
    else:
        return test_sentence_to_label

#######################################
############ triplet stuff ############
#######################################
# 1:1生成的样例
def generate_triplet(label_to_sentences):
    labels = label_to_sentences.keys()
    label_p, label_n = random.sample(labels, 2)
    anchor, pos = random.sample(label_to_sentences[label_p], 2)
    neg = random.sample(label_to_sentences[label_n], 1)[0]
    return anchor, pos, neg

def generate_triplet_batch(label_to_sentences, train_sentence_to_embedding, device, mb_size=64):

    anchor_list = []
    pos_list = []
    neg_list = []
    for _ in range(mb_size):
        anchor, pos, neg = generate_triplet(label_to_sentences)
        anchor_list.append(train_sentence_to_embedding[anchor])
        pos_list.append(train_sentence_to_embedding[pos])
        neg_list.append(train_sentence_to_embedding[neg])

    anchor_embeddings = torch.tensor(anchor_list)
    pos_embeddings = torch.tensor(pos_list)
    neg_embeddings = torch.tensor(neg_list)

    return anchor_embeddings.to(device), pos_embeddings.to(device), neg_embeddings.to(device)

def generate_siamese(label_to_sentence, p):
    labels = label_to_sentence.keys()
    rand_num = random.randint(1, 1+p)
    # 正负样例 p : 1
    # 正样例筛选
    if rand_num == 1:
        label_p = random.sample(labels, 1)[0]
        sentence_1, sentence_2 = random.sample(label_to_sentence[label_p], 2)
        label = 1
    else:
        label_p, label_n = random.sample(labels, 2)
        sentence_1, sentence_2 = random.sample(label_to_sentence[label_p], 1)[0], random.sample(label_to_sentence[label_n], 1)[0]
        label = 0

    return sentence_1, sentence_2, label


def generate_siamese_batch(train_label_to_sentences, train_sentence_to_encoding, device, mb_size=64, p=1):
    sentence_1_list = []
    sentence_2_list = []
    labels_list = []
    for _ in range(mb_size):
        sentence_1, sentence_2, label = generate_siamese(train_label_to_sentences, p)
        sentence_1_list.append(train_sentence_to_encoding[sentence_1])
        sentence_2_list.append(train_sentence_to_encoding[sentence_2])
        labels_list.append(label)
    sentence_1_embeddings = torch.tensor(sentence_1_list)
    sentence_2_embeddings = torch.tensor(sentence_2_list)
    labels = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)

    return sentence_1_embeddings.to(device), sentence_2_embeddings.to(device), labels.to(device)




def load_data(cfg):

    train_sentence_to_label, train_label_to_sentences = get_sentence_to_label(cfg.train_path)
    test_sentence_to_label, _ = get_sentence_to_label(cfg.test_path)

    train_sentence_to_encoding = bert_encoding.get_encoding_dict(train_sentence_to_label, cfg.train_path)
    test_sentence_to_encoding = bert_encoding.get_encoding_dict(test_sentence_to_label, cfg.test_path)
    test_sentence_to_label = get_test_subset(test_sentence_to_label, cfg.val_subset, cfg.seed_num)

    if cfg.samples_per_class:
        train_label_to_sentences = get_train_subset(train_label_to_sentences, cfg.samples_per_class, cfg.seed_num)


    return train_sentence_to_label, train_label_to_sentences, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding


def get_train_x_y(cfg, train_label_to_sentences, train_sentence_to_encoding):
    num_samples = len(list(itertools.chain.from_iterable(train_label_to_sentences.values())))

    train_x = np.zeros((num_samples, cfg.encoding_size))
    train_y = np.zeros((num_samples,))

    i = 0
    for train_label, sentences in train_label_to_sentences.items():
        for sentence in sentences:
            train_x[i, :] = train_sentence_to_encoding[sentence]
            train_y[i] = train_label
            i += 1

    return train_x, train_y


def get_test_x_y(cfg, test_sentence_to_label, test_sentence_to_encoding):
    test_x = np.zeros((len(test_sentence_to_label), cfg.encoding_size))
    test_y = np.zeros((len(test_sentence_to_label),))

    for i, (test_sentence, label) in enumerate(test_sentence_to_label.items()):
        test_x[i, :] = test_sentence_to_encoding[test_sentence]
        test_y[i] = label

    return test_x, test_y