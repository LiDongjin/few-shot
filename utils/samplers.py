import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
import random
from models.sub_layer import bert_encoding

class DataSet(Dataset):
    def __init__(self, file_path, data_type='jd'):
        """
        :param file_path: path to the data
        :param data_type: 'jd' or 'ams'
        """
        self.sentences = []
        self.labels = []
        self.cids = {}
        label = -1  # 将cid转换为label，从0开始.
        lines = open(file_path, 'r', encoding='utf8').readlines()
        if data_type == 'ams':
            lines = lines[1:]

        # 读取每一条数据，将类目转换为类别
        for line in lines:
            parts = line.strip().split('\t')
            cid = int(parts[1].split(',')[-1])
            sentence = parts[0]
            self.sentences.append(sentence)
            if cid not in self.cids:
                label += 1
                self.cids[cid] = label
            self.labels.append(self.cids[cid])

    def __len__(self):
        return len(self.sentences)
    # sentence 与 label 一一对应
    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


class CategoriesSampler(Sampler):

    def __init__(self, sentence_encoding, sentence, label, n_batch, n_cls, n_shot, n_query):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_shot + n_query
        self.label = label  # list
        label = np.array(label)  # array
        self.label_to_index = []
        self.sentence_encoding = sentence_encoding
        self.sentence = sentence
        # 按照类别分组index, 得到每个类别的索引
        for i in range(max(label) + 1):
            idx = np.argwhere(label == i).reshape(-1).tolist()
            self.label_to_index.append(idx)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch_embedding = []
            batch_label = []
            classes = random.sample(self.label, self.n_cls)  # 随机选择类别
            for label in classes:
                indexs = self.label_to_index[label]
                samples = random.sample(indexs, self.n_per)   # 随机选择样本, shot + query 个

                batch.append((self.sentence_encoding[self.sentence[samples]]))
            batch = torch.stack(batch).t().reshape(-1)  #
            yield batch




if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataset = DataSet('C:/Users/johndjli/PycharmProjects/jd-few-shot/data/jd_few_shot_data/10_shot.txt')
    bencoding = bert_encoding.get_encoding_dict(dataset.sentences, 'C:/Users/johndjli/PycharmProjects/jd-few-shot/data/jd_few_shot_data/10_shot.txt')
    sampler = CategoriesSampler(encoding, dataset.sentences, dataset.laels, n_batch=4, n_cls=4, n_shot=5, n_query=1)
    dataloader = DataLoader(dataset=dataset, sampler=sampler)
    for batch in tqdm(dataloader):
        print(batch)