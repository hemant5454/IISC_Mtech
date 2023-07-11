import pickle
import torch

# Open the pickle file for reading
with open('my_file.pkl', 'rb') as f:
    # Read the first object from the file
    obj = pickle.load(f)

second_tuple = ([], [], [], [], [])
for _ in range(len(obj[0])):
    a = torch.zeros((1,17,60))
    b = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    c = torch.zeros((1,17,17))
    d = torch.zeros((1,17,17))
    e = torch.zeros((1,34,34))
    second_tuple[0].append(a)
    second_tuple[1].append(b)
    second_tuple[2].append(c)
    second_tuple[3].append(d)
    second_tuple[4].append(e)

with open('data/second_file.pkl', 'wb') as f:
    # Serialize and write the tuple to the file
    pickle.dump(second_tuple, f)
f.close()