import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# torch.manual_seed(12)
data = [torch.tensor([16]),
        torch.tensor([204, 185, 65, 345]), 
        torch.tensor([204, 987]),
        torch.tensor([85, 4, 23]),
        torch.tensor([125,76,932,5])]

lengths = [d.size(0) for d in data]
# print(lengths)

padded_data = pad_sequence(data, batch_first=True, padding_value=0)
print(padded_data)
mask = (padded_data != 0).float()
print(mask.sum(1))
embedding = nn.Embedding(1000, 128, padding_idx=0)
embeded_data = embedding(padded_data)
print(embeded_data.shape)
# packed_data = pack_padded_sequence(embeded_data, lengths, batch_first=True, enforce_sorted=False)
# unpacked_o, unpacked_lengths = pad_packed_sequence(packed_data, batch_first=True)
# print(packed_data,"\nAAAAAAA")
# print(unpacked_o)
cnn = nn.Conv1d(128, 300, kernel_size=3)
cnn1 = nn.Conv1d(128, 300, kernel_size=4)
embeded_data = embeded_data.permute(0, 2, 1)
print(embeded_data, embeded_data.shape)

# (h, c) is the needed final hidden and cell state, with index already restored correctly by cnn.
# but o is a PackedSequence object, to restore to the original index:

# unpacked_o, unpacked_lengths = pad_packed_sequence(o, batch_first=True)
# now unpacked_o, (h, c) is just like the normal output you expected from a cnn layer.

# print(f'{unpacked_o}\nBBBBB\n {unpacked_lengths}')
fc = nn.Linear(600,2)
# softmax = nn.LogSoftmax(dim=1)
softmax = nn.Softmax(dim=1)
criterion = nn.BCEWithLogitsLoss()

# out = fc(h)
# print(out, len(out))
# o = fc(unpacked_o)
# print(o, len(o))
# a = o.argmax(1)
# print(a)
# out = h.transpose(0,1)
# print(h[-1])
out1 = F.relu(cnn(embeded_data))
out2 = F.relu(cnn1(embeded_data))
print(out1.shape, out2.shape)
out1 = F.max_pool1d(out1, out1.size(2)).squeeze(2)
out2 = F.max_pool1d(out2, out2.size(2)).squeeze(2)
print(out1.shape, out2.shape)
combined = torch.cat((out1, out2), dim=1)
print(combined.shape)
out = fc(combined)
# out = softmax(out)
print(out, out.shape)

# i, res = torch.max(out, 1)
# print(res)
# print([res == torch.tensor([[1],[0],[1],[0]], dtype=torch.float).squeeze(1)])
label = nn.functional.one_hot(torch.tensor([1,0,1,0,1]), num_classes=2).float()
loss = criterion(out, label)
print(label)
print(loss)
probabilities = softmax(out)
predicted_classes = torch.argmax(probabilities, dim=1)
true_classes = torch.argmax(label, dim=1)

print(probabilities)  # Print the probabilities
print(predicted_classes==true_classes)  # Print the predicted class indices