import json

with open("data/train.json", "r", encoding="utf8") as f:
    data = json.load(f)
with open("data/test.json", "r", encoding="utf8") as f:
    data += json.load(f)
sentences = [d["sentence"] for d in data]
corpus_set = set([word for s in sentences for word in s.split()])
print("Corpus size:", len(corpus_set))

word_vectors = []

with open("data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5", "r", encoding="utf8") as f:
    for idx, line in enumerate(f.readlines()):
        if idx == 0:
            continue
        word, vec = line.strip().split(maxsplit=1)
        if word in corpus_set:
            word_vectors.append({"word": word, "vec": list(map(float, vec.split()))})

print("Found vectors:", len(word_vectors))
with open("data/word_vec.json", "w", encoding="utf8") as f:
    json.dump(word_vectors, f, ensure_ascii=False)