import torch
import torch.nn as nn
from utils import dataloader, visualization
import torch.optim as optim
from tqdm import tqdm
from scipy.spatial import distance
from pathlib import Path




def train_mlp(
        cfg
):
    # load data
    train_sentence_to_label, train_label_to_sentences, test_sentence_to_label, train_sentence_to_encoding, test_sentence_to_encoding \
        = dataloader.load_data(cfg)
    train_x, train_y = dataloader.get_train_x_y(cfg, train_label_to_sentences, train_sentence_to_encoding)
    test_x, test_y = dataloader.get_test_x_y(cfg, test_sentence_to_label, test_sentence_to_encoding)

    if cfg.model == "LR":
        model = LR(num_classes=cfg.num_classes)
    else:
        model = MLP(num_classes=cfg.num_classes)

    optimizer = optim.Adam(params=model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)  # wow, works for even large learning rates
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=cfg.decay_gamma)

    num_minibatches_train = int(train_x.shape[0] / cfg.minibatch_size)
    train_loss_list = []
    val_acc_list = []

    ######## training loop ########
    for epoch in range(1, cfg.num_epochs + 1):

        ######## training ########
        model.train()

        train_x, train_y = shuffle(train_x, train_y, random_state=cfg.seed_num)

        for minibatch_num in range(num_minibatches_train):
            start_idx = minibatch_num * cfg.minibatch_size
            end_idx = start_idx + cfg.minibatch_size
            train_inputs = torch.from_numpy(train_x[start_idx:end_idx].astype(np.float32))
            train_labels = torch.from_numpy(train_y[start_idx:end_idx].astype(np.int64))

            # Forward and backpropagation.
            # 局部调整,使得这部分进行梯度计算
            with torch.set_grad_enabled(mode=True):
                train_outputs = model(train_inputs)
                train_conf, train_preds = torch.max(train_outputs, dim=1)
                train_loss = nn.CrossEntropyLoss()(input=train_outputs, target=train_labels)
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        train_loss_list.append(train_loss)

        ######## validation ########
        model.eval()

        val_inputs = torch.from_numpy(test_x.astype(np.float32))
        # val_labels = torch.from_numpy(test_y.astype(np.long))
        val_labels = torch.from_numpy(test_y.astype(np.int64))

        # Feed forward.
        with torch.set_grad_enabled(mode=False):
            val_outputs = model(val_inputs)
            val_confs, val_preds = torch.max(val_outputs, dim=1)
            val_loss = nn.CrossEntropyLoss()(input=val_outputs, target=val_labels)
            val_loss_print = val_loss / val_inputs.shape[0]
            val_acc = accuracy_score(test_y, val_preds)
            val_acc_list.append(val_acc)

    Path(f"plots/{cfg.exp_id}").mkdir(parents=True, exist_ok=True)
    visualization.plot_jasons_lineplot(None, train_loss_list, 'updates', 'training loss',
                                       f"{cfg.exp_id} n_train_c={cfg.samples_per_class} max_val_acc={max(val_acc_list):.3f}",
                                       f"plots/{cfg.exp_id}/train_loss.png")
    visualization.plot_jasons_lineplot(None, val_acc_list, 'updates', 'validation accuracy',
                                       f"{cfg.exp_id} n_train_c={cfg.samples_per_class} max_val_acc={max(val_acc_list):.3f}",
                                       f"plots/{cfg.exp_id}/val_acc{max(val_acc_list):.3f}.png")
    # torch.save(model.state_dict(), 'triplet/models/baseline_covid_weights.pt')


def compute_prototypes():
    return

def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise (ValueError('Unsupported similarity function'))