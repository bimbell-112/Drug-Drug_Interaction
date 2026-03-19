import pandas as pd
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

class DrugInteractionProcessor:
    def __init__(self, max_samples=25000):  # ← UPGRADED: 25K
        self.max_samples = max_samples
        self.drug_encoder = LabelEncoder()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.drug_to_idx = {}
        
    def parse_risk_level(self, description):
        desc = description.lower()
        high_risk = ['increase risk', 'adverse effects', 'toxicity', 'cardiotoxic', 'contraindicated']
        moderate_risk = ['increase', 'decrease', 'may affect', 'serum concentration', 'monitor']
        if any(word in desc for word in high_risk):
            return 2
        elif any(word in desc for word in moderate_risk):
            return 1
        return 0
    
    def smart_sample(self, df, max_samples):
        print(f"📊 Original: {len(df):,} rows")
        df['risk'] = df['Interaction Description'].apply(self.parse_risk_level)
        risk_dist = df['risk'].value_counts()
        print(f"Risk distribution: {risk_dist.to_dict()}")
        
        # Stratified sampling for 25K
        sample_df = df.groupby('risk', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_samples//3), random_state=42)
        ).reset_index(drop=True)
        
        if len(sample_df) > max_samples:
            sample_df = sample_df.sample(max_samples, random_state=42).reset_index(drop=True)
            
        print(f"✅ Reduced to {len(sample_df):,} rows ({len(sample_df)/len(df)*100:.2f}%)")
        return sample_df
    
    def process_data(self, csv_path, embed_dim=384):
        # Load larger chunk for better sampling
        df = pd.read_csv(csv_path, nrows=100000)  # Increased for 25K
        df = self.smart_sample(df, self.max_samples)
        df['risk_level'] = df['Interaction Description'].apply(self.parse_risk_level)
        
        # More drugs for 25K dataset (top 1500)
        all_drugs = pd.concat([df['Drug 1'], df['Drug 2']]).value_counts()
        top_drugs = all_drugs.head(1500).index  # Increased from 1000
        df_filtered = df[df['Drug 1'].isin(top_drugs) & df['Drug 2'].isin(top_drugs)]
        print(f"🧬 Using {len(top_drugs):,} unique drugs")
        
        self.drug_encoder.fit(top_drugs)
        self.drug_to_idx = {drug: i for i, drug in enumerate(top_drugs)}
        num_drugs = len(top_drugs)
        
        src = self.drug_encoder.transform(df_filtered['Drug 1'])
        dst = self.drug_encoder.transform(df_filtered['Drug 2'])
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_labels = torch.tensor(df_filtered['risk_level'].values, dtype=torch.long)
        
        drug_names = list(top_drugs)
        print("🔄 Generating 384-dim embeddings...")
        drug_embeddings = self.embedder.encode(drug_names, batch_size=64, show_progress_bar=True)
        x = torch.tensor(drug_embeddings, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=edge_labels, num_nodes=num_drugs)
        data.edge_label_index = edge_index
        
        os.makedirs('data/processed', exist_ok=True)
        torch.save(data, 'data/processed/drug_graph.pt')
        
        # Train/val/test splits
        train_idx, temp_idx = train_test_split(range(len(edge_labels)), test_size=0.4, 
                                             stratify=edge_labels, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, 
                                           stratify=edge_labels[temp_idx], random_state=42)
        
        pd.DataFrame({'edge_idx': train_idx}).to_csv('data/processed/train.csv', index=False)
        pd.DataFrame({'edge_idx': val_idx}).to_csv('data/processed/val.csv', index=False)
        pd.DataFrame({'edge_idx': test_idx}).to_csv('data/processed/test.csv', index=False)
        
        print(f"✅ 25K GRAPH: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
        return data, (train_idx, val_idx, test_idx)

if __name__ == "__main__":
    processor = DrugInteractionProcessor(max_samples=25000)
    data, splits = processor.process_data('data/raw/db_drug_interactions.csv')
