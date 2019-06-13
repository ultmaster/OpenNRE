import json

with open("../data/nyt/train-all.json", "r") as f:
    data = json.load(f)[:10]
with open("../data/nyt/train.json", "w") as f:
    json.dump(data, f)
with open("../data/nyt/test-all.json", "r") as f:
    data = json.load(f)[:10]
with open("../data/nyt/test.json", "w") as f:
    json.dump(data, f)
