from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='tmp/Cora', name='Cora')

# number of graphs
print("Number of graphs: ", len(dataset))

# number of features
print("Number of features: ", dataset.num_features)

# number of classes
print("Number of classes: ", dataset.num_classes)

# select the first graph
data = dataset[0]

# number of nodes
print("Number of nodes: ", data.num_nodes)

# number of edges
print("Number of edges: ", data.num_edges)

# check if directed
print("Is directed: ", data.is_directed())

# sample nodes from the graph
print("Shape of sample nodes: ", data.x[:5].shape)

# check training nodes
print("# of nodes to train on: ", data.train_mask.sum().item())

# check test nodes
print("# of nodes to test on: ", data.test_mask.sum().item())

# check validation nodes
print("# of nodes to validate on: ", data.val_mask.sum().item())

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import torch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

from layers import GCNLayer, GATLayer

LAYER = "CONV_OWN"
HEADS = 10

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """ GCNConv layers """
        if LAYER == "CONV_TG":
            self.conv1 = GCNConv(data.num_features, 16)
            self.conv2 = GCNConv(16, dataset.num_classes)
        elif LAYER == "CONV_OWN":
            self.conv1 = GCNLayer(data.num_features, 16, use_self_loops=True)
            self.conv2 = GCNLayer(16, dataset.num_classes, use_self_loops=True)
        elif LAYER == "GAT_TG":
            self.conv1 = GATConv(data.num_features, 16, heads=HEADS, concat=True)
            self.conv2 = GATConv(16 * HEADS, dataset.num_classes, concat=False)
        elif LAYER == "GAT_OWN":
            self.conv1 = GATLayer(data.num_features, 16, heads=HEADS, concatenate=True)
            self.conv2 = GATLayer(16 * HEADS, dataset.num_classes, concatenate=False)
        else:
            raise NotImplemented

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# useful function for computing accuracy
def compute_accuracy(pred_y, y):
    return (pred_y == y).sum()

# train the model
model.train()
losses = []
accuracies = []
import time
EPOCHS = 200
t0 = time.time()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    out = model(data)

    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    correct = compute_accuracy(out.argmax(dim=1)[data.train_mask], data.y[data.train_mask])
    acc = int(correct) / int(data.train_mask.sum())
    losses.append(loss.item())
    accuracies.append(acc)

    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print('Epoch: {}, Loss: {:.4f}, Training Acc: {:.4f}'.format(epoch+1, loss.item(), acc))
dt = time.time() - t0
print("------------------------------")
print("total time: %.3f s" % dt)
print("time per epoch: %.3f ms" % (1000 * dt / EPOCHS))
print("time per sample: %.3f ns" % (10e6 * dt / (data.y.shape[0] * EPOCHS)))
# plot the loss and accuracy
"""import matplotlib.pyplot as plt
plt.plot(losses)
plt.plot(accuracies)
plt.legend(['Loss', 'Accuracy'])
plt.show()
"""

# evaluate the model on test set
model.eval()
pred = model(data).argmax(dim=1)
correct = compute_accuracy(pred[data.test_mask], data.y[data.test_mask])
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')