import itertools
import numpy as np
import random
import torch
from models.sub_layer import bert_encoding

def get_sentence_to_label(csv_file):
    lines = open(csv_file, 'r', encoding='utf8').readlines()[1:]
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

# 生产一个batch, support + query
def generate_proto_batch(train_label_to_sentences, train_sentence_to_encoding, device, n_way=5, n_shot=5):
    support_embeddings = []
    query_embeddings = []
    support_label = []
    query_label = []
    labels = list(train_label_to_sentences.keys())
    sample_class = random.sample(labels, n_way)
    cids = {}
    label = -1
    for cid in sample_class:
        if cid not in cids:
            label += 1
            cids[cid] = label
        support_sentence = random.sample(train_label_to_sentences[cid], n_shot)
        query_sentence = [s for s in train_label_to_sentences[cid] if s not in support_sentence]
        for sentence in support_sentence:
            support_embeddings.append(train_sentence_to_encoding[sentence])
            support_label.append(cids[cid])
        for sentence in query_sentence:
            query_embeddings.append(train_sentence_to_encoding[sentence])
            query_label.append(cids[cid])
    return torch.tensor(support_embeddings).to(device), torch.tensor(support_label).to(device), \
           torch.tensor(query_embeddings).to(device), torch.tensor(query_label).to(device)


# def generate_matching_batch(train_label_to_sentences, train_sentence_to_encoding, device, mb_size=64):
#     sentence_embedding = []
#     label = []
#     for _ in range(mb_size):





def load_data(cfg):
    train_sentence_to_label, train_label_to_sentences = get_sentence_to_label(cfg.train_path)
    test_sentence_to_label, _ = get_sentence_to_label(cfg.test_path)

    train_sentence_to_encoding = bert_encoding.get_encoding_dict(train_sentence_to_label, cfg.train_path)
    test_sentence_to_encoding = bert_encoding.get_encoding_dict(test_sentence_to_label, cfg.test_path)
    test_sentence_to_label = get_test_subset(test_sentence_to_label, cfg.val_subset, cfg.seed_num)

    if cfg.samples_per_class:
        train_label_to_sentences = get_train_subset(train_label_to_sentences, cfg.samples_per_class, cfg.seed_num)


    return train_sentence_to_label, train_label_to_sentences, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding

# BERT-encoding 后直接测试的
# def load_test_data(cfg):
#     test_sentence_to_label, test_label_to_sentence = get_sentence_to_label(cfg.test_path)
#     test_sentence_to_encoding = bert_encoding.get_encoding_dict(test_sentence_to_label, cfg.test_path)
#     proto_sentence_to_label = {}
#     query_sentence_to_label = {}
#     proto_label_to_sentence = {}
#     for label, sentences in test_label_to_sentence.items():
#         # 随机打乱
#         random.seed(cfg.seed_num)
#         random.shuffle(sentences)
#         for i in range(5):
#             proto_sentence_to_label[sentences[i]] = label
#             if label not in proto_label_to_sentence:
#                 proto_label_to_sentence[label] = [sentences[i]]
#             else:
#                 proto_label_to_sentence[label].append(sentences[i])
#         for i in range(5, len(sentences)):
#             query_sentence_to_label[sentences[i]] = label
#
#
#     return proto_sentence_to_label, proto_label_to_sentence, query_sentence_to_label, test_sentence_to_encoding, test_sentence_to_encoding


def load_test_data(cfg, istrain=False):
    test_sentence_to_label, test_label_to_sentence = get_sentence_to_label(cfg.test_path)
    test_sentence_to_encoding = bert_encoding.get_encoding_dict(test_sentence_to_label, cfg.test_path)
    proto_sentence_to_label = {}
    query_sentence_to_label = {}
    proto_label_to_sentence = {}

    for label, sentences in test_label_to_sentence.items():
        # 随机打乱
        random.seed(cfg.seed_num)
        random.shuffle(sentences)
        # 截断点, 前面是训练集, 后面是测试集
        if istrain == False:
            mid = 5
        else:
            mid = 8
        for i in range(mid):
            proto_sentence_to_label[sentences[i]] = label
            if label not in proto_label_to_sentence:
                proto_label_to_sentence[label] = [sentences[i]]
            else:
                proto_label_to_sentence[label].append(sentences[i])
        for i in range(mid, len(sentences)):
            query_sentence_to_label[sentences[i]] = label


    return proto_sentence_to_label, proto_label_to_sentence, query_sentence_to_label, test_sentence_to_encoding, test_sentence_to_encoding