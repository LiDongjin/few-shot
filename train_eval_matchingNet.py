from utils import dataloader, visualization
from models.metric.matching_net import MatchingNetwork
from models.sub_layer.embedding import EmbeddingNet
import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_distances
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn.functional as F


def initialize_model(cfg):
    embedding_net = EmbeddingNet(cfg.encoding_size, cfg.embedding_size, batchnorm=True)
    model = MatchingNetwork(embedding_net)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, optimizer, device

def attentionalClassify(similarities, support_set_y, n_way):
    """

    :param similarities: [len_query, len_support]
    :param support_set_y: [len_support, c_cls]
    :param n_way:
    :return:
    """
    softmax_similarities = F.softmax(similarities, dim=-1)  # [query, batch]
    support_set_y_one_hot = torch.zeros(len(support_set_y), n_way).scatter_(1, support_set_y.unsqueeze(1), 1)   # [batch, n_way]
    preds = torch.mm(softmax_similarities, support_set_y_one_hot)   # [query, n_way]
    return preds


def eval_model(
        model,
        device,
        train_sentence_to_label,
        train_label_to_sentences,
        train_sentence_to_encoding,
        test_sentence_to_label,
        test_sentence_to_encoding,
        isTrain=False,
        n_way=5
):
    model.eval()

    def get_sentence_embedding(model, sentence_to_encoding, sentence):
        return model.get_embedding(torch.tensor(sentence_to_encoding[sentence]).to(device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()

    def get_sentence_to_embedding(model, sentence_to_encoding, sentence_to_label, isTrain=False):
        sentence_embedding = []
        sentence_label = []
        cids = {}
        label = -1
        for sentence, cid in sentence_to_label.items():
            if isTrain == False:
                embedding = sentence_to_encoding[sentence]    # 直接用BERT
            else:
                embedding = get_sentence_embedding(model, sentence_to_encoding, sentence)
            sentence_embedding.append(embedding)
            if cid not in cids:
                label += 1
                cids[cid] = label
            sentence_label.append(cids[cid])
        return torch.tensor(sentence_embedding), torch.tensor(sentence_label)

    train_sentence_to_embedding, train_label = get_sentence_to_embedding(model, test_sentence_to_encoding, train_sentence_to_label, isTrain)
    test_sentence_to_embedding, test_label = get_sentence_to_embedding(model, test_sentence_to_encoding, test_sentence_to_label, isTrain)

    distance = pairwise_distances(test_sentence_to_embedding, train_sentence_to_embedding, metric='euclidean')  # [query, support]
    logits = torch.from_numpy(-distance)  # [test_size, support_size]
    preds = attentionalClassify(logits, train_label, n_way)

    values, indices = preds.max(1)
    acc = torch.mean((indices.squeeze() == test_label).float())
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
        isTrain=False,
        n_way=635
    )

    print(f"val_acc={val_acc:.4f}")

def train_eval_model(cfg):
    # load data
    train_sentence_to_label, train_label_to_sentences, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding = dataloader.load_test_data(cfg, istrain=True)

    # initialize model
    model, optimizer, device = initialize_model(cfg)

    # train the model
    iter_bar = tqdm(range(cfg.total_updates))
    update_num_list = []
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []

    Path(f"plots/{cfg.exp_id}").mkdir(parents=True, exist_ok=True)
    writer = open(f"plots/{cfg.exp_id}/logs.csv", "w")
    n_way = 5
    n_shot = 1

    for update_num in iter_bar:
        # load bert batch encoding
        support_embeddings, support_label, query_embeddings, query_label = \
            dataloader.generate_proto_batch(train_label_to_sentences, train_sentence_to_encoding, device, n_way, n_shot)

        model.train()
        model.zero_grad()
        print(model)
        # encoding -> embedding
        support_embeddings = model.get_embedding(support_embeddings)    # [n_way * n_shot, emb_size]
        query_embeddings = model.get_embedding(query_embeddings)    # [n_query, emb_size]

        distance = pairwise_distances(query_embeddings.clone().detach().numpy(), support_embeddings.clone().detach().numpy(), metric='euclidean')    # [query, support]
        logits = torch.from_numpy(-distance).requires_grad_()
        preds = attentionalClassify(logits, support_label, n_way)

        values, indices = preds.max(1)
        acc = torch.mean((indices.squeeze() == query_label).float())
        train_loss = F.cross_entropy(preds, query_label)

        # acc = accuracy_score(query_label, predict_label)
        train_acc_list.append(acc)
        optimizer.zero_grad()
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
                isTrain=True,
                n_way=635
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
        "config/ams_matchingNet/5_shot.json",
    ]

    for cfg_json in cfg_json_list:
        cfg = configuration.triplet_config.from_json(cfg_json)
        print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        # test_model(cfg)
        train_eval_model(cfg)
