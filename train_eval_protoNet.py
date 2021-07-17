from utils import dataloader, visualization
from models.metric.proto import ProtoNet
from models.sub_layer.embedding import EmbeddingNet
import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_distances
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn.functional as F


'''
    1. 本场景下，相当于模型已经训练好，直接进行测试，使用BERT编码，然后计算在测试的类别中，计算准确率，应该效果与KNN差距不会太大。
    2. 可以实验看一下N-WAY训练出来的模型在测试的类别上的效果。
'''
def initialize_model(cfg):
    embedding_net = EmbeddingNet(cfg.encoding_size, cfg.embedding_size, batchnorm=True)
    model = ProtoNet(embedding_net)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, optimizer, device


def eval_model(
        model,
        device,
        train_sentence_to_label,
        train_label_to_sentences,
        train_sentence_to_encoding,
        test_sentence_to_label,
        test_sentence_to_encoding,
        isTrain=False
):
    model.eval()

    def get_sentence_embedding(model, sentence_to_encoding, sentence):
        return model.get_embedding(torch.tensor(sentence_to_encoding[sentence]).to(device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()

    def get_sentence_to_embedding(model, sentence_to_encoding, sentence_to_label, isTrain=False):
        sentence_embedding = []
        sentence_label = []
        for sentence, label in sentence_to_label.items():
            if isTrain == False:
                embedding = sentence_to_encoding[sentence]    # 直接用BERT
            else:
                embedding = get_sentence_embedding(model, sentence_to_encoding, sentence)
            sentence_embedding.append(embedding)
            sentence_label.append(label)
        return sentence_embedding, sentence_label

    def get_proto(model, train_sentence_to_encoding, train_label_to_sentences,isTrain=False):
        proto_embedding = []
        proto_label = []

        for label, sentences in train_label_to_sentences.items():
            if len(sentences) == 1:
                if isTrain == False:
                    embedding = train_sentence_to_encoding[sentences[0]]  # 直接用BERT
                else:
                    embedding = get_sentence_embedding(model, train_sentence_to_encoding, sentence[0])
                proto_embedding.append(embedding)
            else:
                embedding = []
                for sentence in sentences:
                    if isTrain == False:
                        embedding.append(train_sentence_to_encoding[sentence])    # 直接用BERT
                    else:
                        embedding.append(get_sentence_embedding(model, train_sentence_to_encoding, sentence))
                proto_embedding.append(np.mean(embedding, axis=0))
            proto_label.append(label)
        return proto_embedding, proto_label

    proto_embeddings, proto_labels = get_proto(model, train_sentence_to_encoding, train_label_to_sentences, isTrain)
    test_sentence_to_embedding, test_label = get_sentence_to_embedding(model, test_sentence_to_encoding, test_sentence_to_label, isTrain)
    min_index = pairwise_distances_argmin(test_sentence_to_embedding, proto_embeddings, metric='euclidean')

    predicted_label = [proto_labels[x] for x in min_index]
    assert len(predicted_label) == len(test_label)
    num_correct = 0
    for i in range(len(test_label)):
        if predicted_label[i] == test_label[i]:
            num_correct += 1
    acc = num_correct / len(test_label)
    return acc

def test_model(
    cfg
):

    #load data
    train_sentence_to_label, train_label_to_sentences, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding = dataloader.load_test_data(cfg, istrain=False)
    model, optimizer, device = initialize_model(cfg)

    val_acc = eval_model(
        model,
        device,
        train_sentence_to_label,
        train_label_to_sentences,
        train_sentence_to_encoding,
        test_sentence_to_label,
        test_sentence_to_encoding,
        isTrain=False
    )

    print(f"val_acc={val_acc:.4f}")

def train_eval_model(cfg):
    # load data
    train_sentence_to_label, train_label_to_sentences, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding = dataloader.load_test_data(cfg, istrain=True)

    # initialize model
    model, optimizer, device = initialize_model(cfg)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # train the model
    iter_bar = tqdm(range(cfg.total_updates))
    update_num_list = []
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []

    Path(f"plots/{cfg.exp_id}").mkdir(parents=True, exist_ok=True)
    writer = open(f"plots/{cfg.exp_id}/logs.csv", "w")
    n_way = 635
    n_shot = 3

    for update_num in iter_bar:
        # load bert batch encoding
        support_embeddings, support_label, query_embeddings, query_label = \
            dataloader.generate_proto_batch(train_label_to_sentences, train_sentence_to_encoding, device, n_way, n_shot)

        model.train()
        model.zero_grad()

        # encoding -> embedding
        support_embeddings = model.get_embedding(support_embeddings)
        query_embeddings = model.get_embedding(query_embeddings)
        proto = torch.reshape(support_embeddings, [n_way, n_shot, -1]).mean(dim=1).squeeze(1)   # [n_way, emb_dim]
        proto_label = torch.arange(n_way)
        distance = pairwise_distances(query_embeddings.clone().detach().numpy(), proto.clone().detach().numpy(), metric='euclidean')
        logits = torch.from_numpy(-distance).requires_grad_()

        min_index = distance.argmin(axis=1)
        predict_label = [proto_label[x] for x in min_index]

        train_loss = F.cross_entropy(logits, query_label)

        acc = accuracy_score(query_label, predict_label)
        train_acc_list.append(acc)
        train_loss.backward()
        optimizer.step()

        if update_num % cfg.eval_interval == 0:

            val_acc = eval_model(
                model,
                device,
                train_sentence_to_label,
                train_label_to_sentences,
                train_sentence_to_encoding,
                test_sentence_to_label,
                test_sentence_to_encoding,
                isTrain=True
            )
            train_acc = np.mean(train_acc_list)
            train_acc_list = []
            iter_bar_str = (f"update {update_num}/{cfg.total_updates}: "
                            + f"mb_train_loss={float(train_loss):.4f}, "
                            + f"train_acc={float(train_acc):.4f}, "
                            + f"val_acc={float(val_acc):.4f}, "
                            + f"mb_size={n_way*n_shot}, "
                            )
            iter_bar.set_description(iter_bar_str)
            update_num_list.append(update_num)
            val_acc_list.append(val_acc)
            train_loss_list.append(train_loss)
            writer.write(f"{update_num},{val_acc:.4f},{train_loss:.4f}\n")
        if update_num % 2000 == 0:
            lr_scheduler.step()

    visualization.plot_jasons_lineplot(update_num_list, train_loss_list, 'updates', 'training loss',
                                       f"{cfg.exp_id} max_val_acc={max(val_acc_list):.3f}",
                                       f"plots/{cfg.exp_id}/train_loss.png")
    visualization.plot_jasons_lineplot(update_num_list, val_acc_list, 'updates', 'validation accuracy',
                                       f"{cfg.exp_id} max_val_acc={max(val_acc_list):.3f}",
                                       f"plots/{cfg.exp_id}/val_acc{max(val_acc_list):.3f}.png")
    # torch.save(model.state_dict(), 'triplet/models/baseline_covid_weights.pt')


if __name__ == '__main__':
    from utils import common, configuration
    cfg_json_list = [
        # "config/knn/1_shot.json",
        # "config/proto/5_shot.json",
        # "config/proto/10_shot.json",
        # "config/proto/15_shot.json",
        # "config/proto/20_shot.json",
        "config/ams_proto/5_shot.json",
    ]

    for cfg_json in cfg_json_list:
        cfg = configuration.triplet_config.from_json(cfg_json)
        print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        # test_model(cfg)
        train_eval_model(cfg)
