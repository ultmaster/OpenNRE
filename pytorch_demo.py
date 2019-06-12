import json
import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import progressbar
from collections import Counter, defaultdict
import sys

import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_word_vec():
    with open("data/chinese/word_vec.json") as f:
        word_vec_json = json.load(f)
    word2id = {}
    embedding_matrix = np.random.rand(len(word_vec_json) + 2, len(word_vec_json[0]["vec"]))
    embedding_matrix[0] = 0
    for idx, entry in enumerate(word_vec_json, start=1):
        word2id[entry["word"]] = idx
        embedding_matrix[idx] = entry["vec"]
    return word2id, embedding_matrix


def load_relation():
    with open("data/chinese/rel2id.json") as f:
        return json.load(f)


word2id, word_embedding_matrix = load_word_vec()
UNK = len(word2id) + 1
rel2id = load_relation()
max_length = 120


class RelationDataset(Dataset):
    def __init__(self, dataset_name):
        with open("data/chinese/%s.json" % dataset_name) as f:
            data = json.load(f)
        entpair2sent = defaultdict(list)
        entpair2gt = dict()
        for entry in data:
            entpair2sent[(entry["head"]["word"], entry["tail"]["word"])].append(entry["sentence"])
            entpair2gt[(entry["head"]["word"], entry["tail"]["word"])] = rel2id[entry["relation"]]
        self.scope = []
        self.sentences = []
        self.pos1 = []
        self.pos2 = []
        self.lengths = []
        self.relation = []
        self.rel_tot = len(rel2id)
        for (head, tail), sentences in entpair2sent.items():
            current_idx = len(self.sentences)
            for sentence in sentences:
                words = sentence.split()
                self.pos1.append(words.index(head))
                assert self.pos1[-1] != -1
                self.pos2.append(words.index(tail))
                assert self.pos2[-1] != -1
                words_id = [word2id[word] if word in word2id else UNK for word in words]
                words_id += [0] * (max_length - len(words_id))
                self.sentences.append(words_id)
                self.lengths.append(len(words_id))
            self.scope.append([current_idx, current_idx + len(sentences)])
            self.relation.append(entpair2gt[(head, tail)])
        self.relfact_tot = len(self.scope)

        self.weights_table = np.ones(self.rel_tot, dtype=np.float32)
        for r in self.relation:
            self.weights_table[r] += 1
        self.weights_table = 1 / (self.weights_table ** 0.05)

    def __len__(self):
        return len(self.scope)

    def __getitem__(self, idx):
        masks = []
        for i in range(self.scope[idx][0], self.scope[idx][1]):
            mask = np.zeros((3, max_length), dtype=np.float32)
            for j in range(max_length):
                if j <= self.pos1[i]:
                    mask[0][j] = 1
                elif j <= self.pos2[i]:
                    mask[1][j] = 1
                elif j <= self.lengths[i]:
                    mask[2][j] = 1
            masks.append(mask)
        return {
            "bag_size": self.scope[idx][1] - self.scope[idx][0],
            "sentences": self.sentences[self.scope[idx][0]:self.scope[idx][1]],
            "pos1": np.stack(
                [np.arange(max_length) - self.pos1[i] + max_length for i in range(self.scope[idx][0], self.scope[idx][1])]),
            "pos2": np.stack(
                [np.arange(max_length) - self.pos2[i] + max_length for i in range(self.scope[idx][0], self.scope[idx][1])]),
            "masks": np.stack(masks),
            "relation": self.relation[idx],
        }


def relation_collate_fn(batch):
    ret = {"scope": [], "sentences": [], "pos1": [], "pos2": [], "masks": [], "relation": []}
    for b in batch:
        ret["scope"].append([len(ret["sentences"]), len(ret["sentences"]) + b["bag_size"]])
        for key in ["sentences", "pos1", "pos2", "masks"]:
            ret[key].extend(b[key])
        ret["relation"].append(b["relation"])
    assert ret["scope"][-1][1] == len(ret["sentences"])
    return {k: torch.tensor(v) for k, v in ret.items()}


class CNN(nn.Module):

    def __init__(self, selector="ave", encoder="cnn"):
        super(CNN, self).__init__()
        self.num_filters = 230
        self.position_dim = 5
        self.wordvec_dim = 300
        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding_matrix).float(),
                                                            freeze=False)
        self.pf1 = nn.Embedding(2 * max_length, self.position_dim)
        self.pf2 = nn.Embedding(2 * max_length, self.position_dim)
        self.conv = nn.Conv1d(self.wordvec_dim + 2 * self.position_dim, self.num_filters,
                              3, padding=1, padding_mode="border")

        self.selector = selector
        self.encoder = encoder

        if self.encoder == "cnn":
            self.dense = nn.Linear(self.num_filters, len(rel2id))
        elif self.encoder == "pcnn":
            self.dense = nn.Linear(self.num_filters * 3, len(rel2id))
        else:
            raise NotImplementedError

    def logit(self, hidden):
        return F.softmax(self.dense(hidden), dim=1)

    def forward(self, data):
        # embedding
        w_embeddings = self.word_embeddings(data["sentences"])
        pf1_embeddings = self.pf1(data["pos1"])
        pf2_embeddings = self.pf2(data["pos2"])
        embeds = torch.cat((w_embeddings, pf1_embeddings, pf2_embeddings), 2)

        # encoder
        if self.encoder == "cnn":
            conv_result = self.conv(embeds.permute((0, 2, 1)))
            max_val, _ = torch.max(conv_result.view((-1, self.num_filters, max_length)), dim=2)
            pooling_result = F.dropout(F.relu(max_val), p=0.5, training=self.training)
        elif self.encoder == "pcnn":
            conv_result = self.conv(embeds.permute((0, 2, 1)))
            max_val = torch.max(conv_result.unsqueeze(1) + 100 * data["masks"].unsqueeze(2), dim=3)[0] - 100
            pooling_result = F.dropout(F.relu(max_val.view((-1, 3 * self.num_filters))), p=0.5, training=self.training)
            assert pooling_result.size(0) == conv_result.size(0)
        else:
            raise NotImplementedError

        # selector
        if self.selector == "ave":
            selector_result = torch.stack([torch.mean(pooling_result[a:b], 0) for a, b in data["scope"]])
            bag_repre = F.dropout(selector_result, p=0.5, training=self.training)
        elif self.selector == "one":
            if self.training:  # training
                bag_repre = []
                for r, (a, b) in zip(data["relation"], data["scope"]):
                    bag_hidden_mat = pooling_result[a:b]
                    instance_logit = self.dense(bag_hidden_mat)
                    j = torch.argmax(instance_logit[:, r])
                    bag_repre.append(bag_hidden_mat[j])
                bag_repre = F.dropout(torch.stack(bag_repre), p=0.5)
            else:  # testing
                bag_logit = []
                for a, b in data["scope"]:
                    instance_logit = self.logit(pooling_result[a:b])
                    reduced_max, _ = torch.max(instance_logit, 0)
                    bag_logit.append(reduced_max)
                bag_logit = torch.stack(bag_logit)
                return bag_logit
        else:
            raise NotImplementedError

        # classifier
        return self.logit(bag_repre)


parser = ArgumentParser()
parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("--epoch", type=int, help="epoch number", default=120)
parser.add_argument("--batch", type=int, help="batch size", default=16)
parser.add_argument("--optimizer", type=str, help="optimizer", default="adam")
parser.add_argument("--encoder", type=str, help="encoder", default="cnn")
parser.add_argument("--selector", type=str, help="selector", default="ave")
args = parser.parse_args()


train_dataset = RelationDataset("train")
train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=relation_collate_fn)
test_dataset = RelationDataset("test")
test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=True, collate_fn=relation_collate_fn)


net = CNN(selector=args.selector, encoder=args.encoder)
net.to(device)
criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(train_dataset.weights_table))
criterion.to(device)
if args.optimizer == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.optimizer == "rmsprop":
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
elif args.optimizer == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

else:
    raise NotImplementedError

model_name = "torch_%s_%s_%s_%d_%f" % (args.optimizer, args.encoder, args.selector, args.batch, args.lr)

log_file = open("./summary/" + model_name + ".log", "w")
best_accuracy = 0
non_best_stop = 0
best_auc = 0

for epoch in range(args.epoch):
    sys.stdout.flush()
    widgets = ['Epoch %03d [' % epoch, progressbar.Bar(), ' ',
               progressbar.Counter(), '/%d ' % len(train_dataset),
               progressbar.Timer(), ' ', progressbar.ETA(), ' ',
               progressbar.DynamicMessage('loss'), ' ', progressbar.DynamicMessage('acc')]
    bar = progressbar.ProgressBar(max_value=len(train_dataset), widgets=widgets)
    epoch_loss = []
    epoch_tot, epoch_acc = 0, 0
    for i, data_cpu in enumerate(train_dataloader):
        sys.stdout.flush()
        data = {k: v.to(device) for k, v in data_cpu.items()}
        net.train()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(data)
        loss = criterion(outputs, data["relation"])
        loss.backward()
        net.word_embeddings.weight.grad[0] = 0  # zero BLANK grad

        # optimize
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        run_acc = (predicted == data["relation"]).sum().item()
        running_loss = loss.item()

        epoch_loss.append(running_loss)
        epoch_tot += len(data["relation"])
        epoch_acc += run_acc

        bar.update(epoch_tot, loss=running_loss, acc=epoch_acc / epoch_tot)

        sys.stdout.flush()

    print("")  # linebreak

    # evaluate
    net.eval()
    pred_result = []
    with torch.no_grad():
        total, correct = 0, 0
        for i, data_cpu in enumerate(test_dataloader):
            data = {k: v.to(device) for k, v in data_cpu.items()}
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == data["relation"]).sum().item()
            total += len(data["relation"])
            for idx in range(len(outputs)):
                for rel in range(1, test_dataset.rel_tot):
                    pred_result.append({'score': outputs[idx][rel],
                                        'flag': int(data["relation"][idx].item() == rel)})
        accuracy = correct / total

    def log(*args):
        print(*args)
        print(*args, file=log_file)


    log("[Epoch %03d] [%s] loss: %.6f acc: %.6f test: %.6f" % (epoch, datetime.now(),
                                                               sum(epoch_loss) / len(epoch_loss),
                                                               epoch_acc / epoch_tot, accuracy))
    sorted_test_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
    prec, recall = [], []
    correct = 0
    for i, item in enumerate(sorted_test_result):
        correct += item['flag']
        prec.append(float(correct) / (i + 1))
        recall.append(float(correct) / test_dataset.relfact_tot)
    auc = sklearn.metrics.auc(x=recall, y=prec)
    log("[TEST] auc = %.6f" % auc)
    if auc > best_auc:
        best_auc = auc
        best_accuracy = accuracy
        print("Best model, saving...")
        torch.save(net, "./checkpoint/%s_%03d.pkl" % (model_name, epoch))
        np.save(os.path.join("./test_result", model_name + "_x.npy"), np.array(recall))
        np.save(os.path.join("./test_result", model_name + "_y.npy"), np.array(prec))
        non_best_stop = 0
    else:
        non_best_stop += 1

    if non_best_stop >= 20:
        print("Accuracy hasn't improved in the last 20 epochs, exiting...")
        break

    sys.stdout.flush()

print("Best accuracy: %.6f" % best_accuracy, file=log_file)
log_file.close()
