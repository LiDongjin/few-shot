from utils import common
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from pathlib import Path
import os


model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'C:/Users/johndjli/PycharmProjects/jd-few-shot/jd_8.6kw_model_only_bert/'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

def get_encodings_path(file_path):
    encodings_path = Path(file_path).parent.joinpath(file_path.split('/')[-1].split('.')[0] + '_encodings.pkl')
    return encodings_path

# 将每句话的embedding编码保存到pkl文件，并返回
# def get_encoding_dict(sentence_to_labels, file_path):
#     encodings_path = get_encodings_path(file_path)
#     if not encodings_path.exists():
#         print(f"creating {encodings_path}")
#         string_to_encoding = {}
#
#         for sentence in tqdm(sentence_to_labels.keys()):
#             encoding = avg_Pooling(sentence, tokenizer, model)
#             string_to_encoding[sentence] = encoding
#
#         common.save_pickle(encodings_path, string_to_encoding)
#
#     return common.load_pickle(encodings_path)

def get_encoding_dict(sentences, file_path):
    encodings_path = get_encodings_path(file_path)
    if not encodings_path.exists():
        print(f"creating {encodings_path}")
        string_to_encoding = {}

        for sentence in tqdm(sentences):
            encoding = avg_Pooling(sentence, tokenizer, model)
            string_to_encoding[sentence] = encoding

        common.save_pickle(encodings_path, string_to_encoding)

    return common.load_pickle(encodings_path)

# Encode text
def avg_Pooling(input_string, tokenizer, model):
    input_ids = torch.tensor([tokenizer.encode(input_string, add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0].numpy()  # Models outputs are now tuples
        # last_hidden_states = last_hidden_states[:, 0, :]
        last_hidden_states = np.mean(last_hidden_states, axis=1)
        last_hidden_states = last_hidden_states.flatten()
        return last_hidden_states
