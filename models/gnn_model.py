import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DrugInteractionGNN(torch.nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_classes=3):  # Larger hidden
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim//2)
        self.conv4 = GCNConv(hidden_dim//2, 128)  # Extra layer
        
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(256, 128),  # Increased capacity
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, num_classes)
        )
        
    def forward(self, x, edge_index, edge_label_index=None):
        # Deeper GNN
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        
        if edge_label_index is not None:
            src, tgt = edge_label_index
            edge_emb = torch.cat([x[src], x[tgt]], dim=-1)
            return self.edge_mlp(edge_emb)
        return x

def predict_interaction(model, data, drug1_idx, drug2_idx):
    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.edge_index)
        edge_emb = torch.cat([emb[drug1_idx], emb[drug2_idx]], dim=-1)
        logits = model.edge_mlp(edge_emb.unsqueeze(0))
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(1).item()
    return pred, probs.numpy()
