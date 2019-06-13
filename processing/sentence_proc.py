import json
from collections import defaultdict

import jieba


with open("data/people.txt", "r", encoding="utf8") as f:
    entities = list(filter(lambda s: s, map(lambda s: s.strip(), f.readlines())))
entities_dict = dict()
for i, entity in enumerate(entities, start=1):
    entities_dict[entity] = "m.%03d" % i

relation_dict = {"NA": 0}

for word in entities_dict.keys():
    jieba.add_word(word)
sentence_cut = []
entity2sent = defaultdict(list)

with open("data/sentence.txt", "r", encoding="utf8") as f:
    for sentence in f.readlines():
        s = sentence.strip()
        if not s:
            continue
        for w in ["[1]", "[2]", "[3]", "[4]", "[5]"]:
            s = s.replace(w, "")
        words = jieba.lcut(s)
        sentence_cut.append(words)
        assert len(words) <= 100
        idx = len(sentence_cut) - 1
        for word in words:
            if word in entities_dict:
                entity2sent[word].append(idx)

entity2sent = {k: set(v) for k, v in entity2sent.items()}


def go(filename):
    ret = []
    tot = success = 0
    with open("data/" + filename + "_relation.txt", "r", encoding="utf8") as f:
        for line in f.readlines():
            a, b, c = line.strip().split("\t")
            if c not in relation_dict:
                sz_relation = len(relation_dict)
                relation_dict[c] = sz_relation
            tot += 1
            if a not in entity2sent or b not in entity2sent or a == b:
                continue
            success += 1
            for sent_idx in entity2sent[a] & entity2sent[b]:
                ret.append({
                    "sentence": " ".join(sentence_cut[sent_idx]),
                    "head": {"word": a, "id": entities_dict[a]},
                    "tail": {"word": b, "id": entities_dict[b]},
                    "relation": c
                })
    print(filename, success, "out of", tot, "succeed")
    # ret = ret[:2]
    with open("data/" + filename + ".json", "w", encoding="utf8") as f:
        json.dump(ret, f, ensure_ascii=False)
    with open("data/" + filename + ".pretty.json", "w", encoding="utf8") as f:
        json.dump(ret, f, indent=2, sort_keys=True, ensure_ascii=False)


go("train")
go("test")
with open("data/rel2id.json", "w", encoding="utf8") as f:
    json.dump(relation_dict, f, indent=2, sort_keys=True, ensure_ascii=False)
