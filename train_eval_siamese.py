from utils import dataloader, visualization
from models.metric.siamese import SiameseNet
from models.sub_layer.embedding import EmbeddingNet
import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import pairwise_distances_argmin
from transformers import AdamW

def initialize_model(cfg):
    embedding_net = EmbeddingNet(cfg.encoding_size, cfg.embedding_size)
    model = SiameseNet(embedding_net, cfg.embedding_size)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

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
):
    model.eval()

    def get_sentence_embedding(model, sentence_to_encoding, sentence):
        return model.get_embedding(torch.tensor(sentence_to_encoding[sentence]).to(device)).detach().cpu().numpy()

    def get_sentence_to_embedding(model, sentence_to_encoding, sentence_to_label):
        sentence_embedding = []
        sentence_label = []
        for sentence, label in sentence_to_label.items():
            embedding = get_sentence_embedding(model, sentence_to_encoding, sentence)
            sentence_embedding.append(embedding)
            sentence_label.append(label)
        return sentence_embedding, sentence_label

    train_sentence_to_embedding, train_label = get_sentence_to_embedding(model, train_sentence_to_encoding,
                                                                         train_sentence_to_label)
    test_sentence_to_embedding, test_label = get_sentence_to_embedding(model, test_sentence_to_encoding,
                                                                       test_sentence_to_label)
    min_index = pairwise_distances_argmin(test_sentence_to_embedding, train_sentence_to_embedding, metric='l1')

    predicted_label = [train_label[x] for x in min_index]
    assert len(predicted_label) == len(test_label)
    num_correct = 0
    for i in range(len(test_label)):
        if predicted_label[i] == test_label[i]:
            num_correct += 1
    acc = num_correct / len(test_label)
    return acc


def train_eval_model(cfg):
    # load data
    train_sentence_to_label, train_label_to_sentences, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding \
        = dataloader.load_data(cfg)

    # initialize model
    model, optimizer, device = initialize_model(cfg)

    # train the model
    iter_bar = tqdm(range(cfg.total_updates))
    update_num_list = [];
    train_loss_list = [];
    val_acc_list = []

    Path(f"plots/{cfg.exp_id}").mkdir(parents=True, exist_ok=True)
    writer = open(f"plots/{cfg.exp_id}/logs.csv", "w")
    mb_size = 64

    for update_num in iter_bar:
        # batch embeddings of anchor, pos, neg
        sentence_1, sentence_2, labels = dataloader.generate_siamese_batch(train_label_to_sentences, train_sentence_to_encoding,
                                                             device, mb_size=mb_size, p=1)
        model.train()
        model.zero_grad()
        train_loss = model(sentence_1, sentence_2, labels)

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
            )


            iter_bar_str = (f"update {update_num}/{cfg.total_updates}: "
                            + f"mb_train_loss={float(train_loss):.4f}, "
                            + f"val_acc={float(val_acc):.4f}, "
                            + f"mb_size={mb_size}"
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
        "config/siamese/5_shot.json",
        "config/siamese/10_shot.json",
        "config/siamese/15_shot.json",
        "config/siamese/20_shot.json",
    ]

    for cfg_json in cfg_json_list:
        cfg = configuration.triplet_config.from_json(cfg_json)
        print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        train_eval_model(cfg)
