import torch
import torch.nn.functional as F
from models.gnn_model import DrugInteractionGNN
from utils.data_processor import DrugInteractionProcessor
import os

def train():
    print("🚀 25K Drug Interaction GNN Training...")
    
    processor = DrugInteractionProcessor(max_samples=25000)  # 25K
    data, splits = processor.process_data('data/raw/db_drug_interactions.csv')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    print(f"📈 Graph: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    
    model = DrugInteractionGNN().to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    best_loss = float('inf')
    for epoch in range(80):  # More epochs for 25K
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_label_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}, Loss: {loss.item():.4f}")
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'best_model_25k.pt')
            print(f"💾 Best model: {best_loss:.4f}")
    
    print("✅ 25K Training COMPLETE!")
    print(f"📁 Model saved: best_model_25k.pt")
    return model

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    train()
