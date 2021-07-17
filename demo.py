import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.sub_layer import bert_encoding
from utils.samplers import jdDataSet, CategoriesSampler
from models.metric.proto import ProtoNet
from sklearn.metrics import pairwise_distances_argmin, accuracy_score
from utils import dataloader

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file-path', type=str, default='data/jd_few_shot_data/1_shot.txt')
    parser.add_argument('--valid-file-path', type=str, default='data/jd_few_shot_data/JD_test.txt')
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=256)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/proto-1')
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=400)
    parser.add_argument('--learning', type=float, default=0.001)
    args = parser.parse_args()

    # ensure_path(args.save_path)
    train_sentence_to_label, train_label_to_sentences, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding \
        = dataloader.load_data(args)
    trainset = jdDataSet(args.train_file_path)
    train_sampler = CategoriesSampler(trainset.label, args.train_batch_size, args.train_way, args.shot, args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    valset = jdDataSet(args.valid_file_path)
    val_sampler = CategoriesSampler(valset.label, args.test_batch_size, args.test_way, args.shot, args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProtoNet()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),  lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), os.path.join(args.save_path, name + '.pth'))


    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot = data[:p]    # 原型shot
            data_query = data[p:]   # 待标注的样例

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)  # 原型中心

            label = torch.arange(args.train_way).repeat(args.query)  #
            label = label.type(torch.cuda.LongTensor)

            logits = model(data_query)
            predict_label = pairwise_distances_argmin(logits, proto, metric='euclidean')
            loss = F.cross_entropy(logits, label)
            acc = accuracy_score(label, predict_label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = model(data_query)
            predict_label = pairwise_distances_argmin(logits, proto, metric='euclidean')  #
            loss = F.cross_entropy(logits, label)
            acc = accuracy_score(label, predict_label)

            vl.add(loss.item())
            va.add(acc)


        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        # if va > trlog['max_acc']:
        #     trlog['max_acc'] = va
        #     save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        # torch.save(trlog, os.path.join(args.save_path, 'trlog'))

        # save_model('epoch-last')

        # if epoch % args.save_epoch == 0:
        #     save_model('epoch-{}'.format(epoch))

