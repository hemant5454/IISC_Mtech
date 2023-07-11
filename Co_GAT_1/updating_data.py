import json
# /data/home/hemantmishra/jaydeep_model/Co-GAT/dataset/mastodon/train.json
with open('/data/home/hemantmishra/jaydeep_model/Co-GAT/dataset/mastodon/train.json') as f:
    train = json.load(f)
f.close()
with open('/data/home/hemantmishra/jaydeep_model/Co-GAT/dataset/mastodon/test.json') as f:
    test = json.load(f)
f.close()
with open('/data/home/hemantmishra/jaydeep_model/Co-GAT/dataset/mastodon/dev.json') as f:
    dev = json.load(f)
f.close()

# print('x')
# sentiments = []
# acts = []
# length = len(data)
max_length1 = max([len(y) for y in train])
max_length2 = max([len(y) for y in test])
max_length3 = max([len(y) for y in dev])
max_length = max(max_length1, max_length2, max_length3)
# for x in data:
#     for y in x:
#         sentiments.append(y['sentiment'])
#         acts.append(y['act'])
# a = list(set(sentiments))
# b = list(set(acts))
# b.sort()
pad = {'utterance': '', 'sentiment': '*', 'act': 'L'}
for batch in train:
    pad_length = max_length - len(batch)
    for _ in range(pad_length):
        batch.append(pad)
for batch in test:
    pad_length = max_length - len(batch)
    for _ in range(pad_length):
        batch.append(pad)
for batch in dev:
    pad_length = max_length - len(batch)
    for _ in range(pad_length):
        batch.append(pad)


with open('/data/home/hemantmishra/jaydeep_model/Co-GAT/dataset/mastodon_updated/train.json', 'w') as f:
    json.dump(train, f)
f.close()
with open('/data/home/hemantmishra/jaydeep_model/Co-GAT/dataset/mastodon_updated/test.json', 'w') as f:
    json.dump(test, f)
f.close()
with open('/data/home/hemantmishra/jaydeep_model/Co-GAT/dataset/mastodon_updated/dev.json', 'w') as f:
    json.dump(dev, f)
f.close()

with open('/data/home/hemantmishra/jaydeep_model/Co-GAT/dataset/mastodon/train_adj.txt') as f:
    x = json.load(f)
f.close()
with open('/data/home/hemantmishra/jaydeep_model/Co-GAT/dataset/mastodon/test_adj.txt') as f:
    y = json.load(f)
f.close()
with open('/data/home/hemantmishra/jaydeep_model/Co-GAT/dataset/mastodon/dev_adj.txt') as f:
    z = json.load(f)
f.close()

with open('/data/home/hemantmishra/jaydeep_model/Co-GAT/dataset/mastodon_updated/train_adj.txt', 'w') as f:
    json.dump(x, f)
f.close()
with open('/data/home/hemantmishra/jaydeep_model/Co-GAT/dataset/mastodon_updated/test_adj.txt', 'w') as f:
    json.dump(y, f)
f.close()
with open('/data/home/hemantmishra/jaydeep_model/Co-GAT/dataset/mastodon_updated/dev_adj.txt', 'w') as f:
    json.dump(z, f)
f.close()
