from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from safetensors.torch import safe_open
import random
from torch.utils.data import Dataset
np.random.seed(1984)

import hashlib

# Global mapping to maintain string-to-int relationships
_string_int_mapping = {}
_int_string_mapping = {}

def string_to_int(s):
    """Convert string to integer using stable hash, maintaining bidirectional mapping"""
    s = str(s)
    if s in _string_int_mapping:
        return _string_int_mapping[s]
    
    # Use SHA-256 hash and take first 4 bytes as integer
    # This ensures deterministic results across sessions
    hash_bytes = hashlib.sha256(s.encode('utf-8')).digest()[:4]
    int_val = int.from_bytes(hash_bytes, 'big') % (2**31)  # Ensure positive 31-bit integer
    
    # Store bidirectional mapping
    _string_int_mapping[s] = int_val
    _int_string_mapping[int_val] = s
    
    return int_val

def int_to_string(n):
    """Convert integer back to string using stored mapping"""
    return _int_string_mapping.get(n, f'cluster_{n}')

class RepeatDataset(Dataset):
    """Dataset wrapper that repeats a dataset multiple times."""
    
    def __init__(self, dataset, repeat=1):
        self.dataset = dataset
        self.repeat = repeat
        
    def __len__(self):
        return len(self.dataset) * self.repeat
    
    def __getitem__(self, idx):
        """Sample from the original dataset by computing idx % len(dataset)."""
        return self.dataset[idx % len(self.dataset)]


class ListDataset(Dataset): # tracks scalers for different data types
    def __init__(self, path_to_tsv, reverse_augmentation=False, reverse=False, indices=None):
        self.path_to_tsv = path_to_tsv
        self.reverse_augmentation = reverse_augmentation
        
        dataframe = pd.read_csv(path_to_tsv, sep="\t")
        self.data = dataframe.dropna(subset=["wt_seq", "mutant_seq"])  # Drop rows with NaN in sequence columns
            
        if "cluster" not in self.data.columns:
            self.data["cluster"] = -1
        else:
            # Use simple string-to-int encoding for cluster names
            for i, row in self.data.iterrows():
                if pd.notna(row["cluster"]):
                    self.data.at[i, "cluster"] = string_to_int(str(row["cluster"]))
                else:
                    self.data.at[i, "cluster"] = -1

        if "wt_uid" in self.data.columns:
            self.data = self.data.drop(columns=["wt_uid"])
        
        if "ddG_ML" in self.data.columns:
            self.labels = self.data["ddG_ML"].astype(np.float32).to_list()
        else:
            self.labels = self.data["ddG"].astype(np.float32).to_list()

        self.ids = self.data.index.tolist()
        self.wt_seq = self.data["wt_seq"].tolist()
        self.mut_seqs = self.data["mutant_seq"].tolist()
        self.cluster = self.data["cluster"].tolist()
        del self.data
        
        if reverse:
            self.wt_seq, self.mut_seqs = self.mut_seqs, self.wt_seq
            self.labels = [-1 * label for label in self.labels]
        
        if indices:  # Leave only specific indices
            self.ids = [self.ids[i] for i in indices]
            self.wt_seq = [self.wt_seq[i] for i in indices]
            self.mut_seqs = [self.mut_seqs[i] for i in indices] 
            self.labels = [self.labels[i] for i in indices]
        
        self.len = len(self.labels)
    

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.reverse_augmentation:
            if random.random() < 0.5:
                return {'wt': self.mut_seqs[idx], 'mut': self.wt_seq[idx], 'labels': -1 * self.labels[idx], "cluster": torch.tensor(self.cluster[idx], dtype=torch.long)}
            else:
                return {'wt': self.wt_seq[idx], 'mut': self.mut_seqs[idx], 'labels': self.labels[idx], "cluster": torch.tensor(self.cluster[idx], dtype=torch.long)}
        else:
            return {'wt': self.wt_seq[idx], 'mut': self.mut_seqs[idx], 'labels': self.labels[idx], "cluster": torch.tensor(self.cluster[idx], dtype=torch.long)}


def weighted_random_list(choices):
    # list is [(item1, weight1), (item2, weight2), ...]
    if not choices:
        return None
    items, weights = zip(*choices)
    return random.choices(items, weights=weights, k=1)[0]


class HierarchicalSyntheticDataset(Dataset):
    """Stochastic sampling of mutations with synthetic mutation generation.
    No pre-generation of synthetic mutations - they're generated on-the-fly
    Stochastic sampling instead of deterministic indexing
    """
    
    def __init__(self, paths_to_tsv, mutation_types = {}, reverse_augmentation=False):
        self.mutation_types = mutation_types # dictionary of mutation types and probabilities.    
        self.reverse_augmentation = reverse_augmentation    
        
        # Load and preprocess data
        dataframes = []
        paths_to_tsv = [paths_to_tsv]
        for path_to_tsv in paths_to_tsv:
            dataframe = pd.read_csv(path_to_tsv, sep="\t")
            dataframes.append(dataframe)
        self.data = pd.concat(dataframes).dropna(subset=["wt_seq", "mutant_seq"])
        
        # Standardize column names
        if "cluster" not in self.data.columns:
            self.data["cluster"] = self.data["megawt_cluster"]
        if "ddG" not in self.data.columns:
            self.data["ddG"] = self.data["ddG_ML"]
            
        # Pre-compute sampling structures for speed
        self._precompute_sampling_structures()

    def _precompute_sampling_structures(self):
        """Pre-compute all sampling structures to avoid repeated computation."""
        # Build hierarchical data structure
        self.hierarchical_data = {} # {cluster: {wt: {mut_type: [(mut1,  label1), (mut2, label2), ...]}}}
        self.cluster_list = []  # Pre-computed list for faster access
        self.wt_lists = {}  # Pre-computed wt lists per cluster
        
        # Pre-compute weighted random sampling structures
        self.mut_type_choices = [mut_type for mut_type, _ in self.mutation_types]
        self.mut_type_weights = [weight for _, weight in self.mutation_types]
    
        
        clusters = self.data["cluster"].unique()
        
        for cluster in clusters:
            cluster_data = self.data[self.data["cluster"] == cluster]
            wt_seqs = cluster_data["wt_seq"].unique()
            
            self.hierarchical_data[cluster] = {}
            self.wt_lists[cluster] = list(wt_seqs)
            
            for wt in wt_seqs:
                wt_data = cluster_data[cluster_data["wt_seq"] == wt]
                
                # Initialize wt in hierarchical data with mutation types
                self.hierarchical_data[cluster][wt] = {}
                
                # Store real mutations by type
                for mut_type in ['single', 'double']:
                    mut_data = wt_data[wt_data["mutation_type"] == mut_type]
                    if not mut_data.empty:
                        mutations = list(zip(
                            mut_data["mutant_seq"].tolist(),
                            mut_data["ddG"].astype(np.float32).tolist()
                        ))   # [(mut1, label1), (mut2, label2), ...]
                        self.hierarchical_data[cluster][wt][mut_type] = mutations

        # AVAILABLE SYNTHETIC MUTATIONS
        # ['zero', 'single', 'double', 'synthetic_double', 'triple', 'quadruple']
        # NOTE: There are intersecting doubles, so quadruple return doubles. Plus doubles and synthetic doubles are sampled equally. RESULT -> a lot of doubles. ~2x
        self.cluster_list = list(self.hierarchical_data.keys())
        
    def __len__(self):
        return len(self.cluster_list)

    def _can_generate_synthetic(self, available_mutations, mut_type):
        """Check if synthetic mutation can be generated without actually generating it."""
        if mut_type == 'synthetic_double':
            return available_mutations.get('single') and len(available_mutations['single']) >= 2
        elif mut_type == 'triple':
            return (available_mutations.get('double') and 
                   available_mutations.get('single'))
        elif mut_type == 'quadruple':
            return (available_mutations.get('double') and 
                   len(available_mutations['double']) >= 2)
        return False

    def _generate_synthetic_mutation(self, available_mutations, mut_type):
        """Generate a synthetic mutation on the fly."""
        if mut_type == 'synthetic_double':
            mut1, label1 = random.choice(available_mutations['single'])
            mut2, label2 = random.choice(available_mutations['single'])
        elif mut_type == 'triple':
            mut1, label1 = random.choice(available_mutations['single'])
            mut2, label2 = random.choice(available_mutations['double'])
        elif mut_type == 'quadruple':
            mut1, label1 = random.choice(available_mutations['double'])
            mut2, label2 = random.choice(available_mutations['double'])
        new_label = -label1 + label2
        if abs(new_label) < 0.5: # small effect mutation
            mut1, mut2 = None, None
            new_label = None
        return mut1, mut2, new_label
    
    def __getitem__(self, idx):
        """Optimized stochastic sampling with retry limit."""
        max_retries = 10  # Prevent infinite loops
        
        for attempt in range(max_retries):
            # Use pre-computed lists for faster access
            cluster = self.cluster_list[idx % len(self.cluster_list)]
            wt = random.choice(self.wt_lists[cluster])
            
            # Fast weighted sampling using pre-computed structures
            if self.mut_type_choices:
                mut_type = random.choices(self.mut_type_choices, 
                                        weights=self.mut_type_weights, k=1)[0]
            else:
                mut_type = 'single'  # fallback
            
            sample_null = (mut_type == 'zero')
            if sample_null:
                # Pick a different mutation type for the basis
                non_zero_choices = [c for c in self.mut_type_choices if c != 'zero']
                non_zero_weights = [w for t, w in self.mutation_types if t in non_zero_choices]
                if non_zero_choices:
                    mut_type = random.choices(non_zero_choices, weights=non_zero_weights, k=1)[0]
                else:
                    mut_type = 'single'  # fallback
            
            available_mutations = self.hierarchical_data[cluster][wt]
            
            # Try to get mutation
            if mut_type in ['single', 'double']:
                if mut_type in available_mutations:
                    mut, label = random.choice(available_mutations[mut_type])
                else:
                    continue  # Retry with different sampling
                    
            else:  # Synthetic mutation
                if self._can_generate_synthetic(available_mutations, mut_type):
                    wt, mut, label = self._generate_synthetic_mutation(available_mutations, mut_type)
                    if mut is None:
                        continue  # Retry
                else:
                    continue  # Retry with different sampling
            
            # Apply transformations
            if self.reverse_augmentation and random.random() < 0.5:
                label = -label
                mut, wt = wt, mut
            
            if sample_null:
                return {'wt': wt, 'mut': wt, 'labels': 0.0, 'cluster': torch.tensor(-1, dtype=torch.long)}
            
            return {'wt': wt, 'mut': mut, 'labels': float(label), 'cluster': torch.tensor(-1, dtype=torch.long)}
        
        # Fallback if all retries failed - return a simple null mutation
        cluster = self.cluster_list[0]
        wt = self.wt_lists[cluster][0]
        return {'wt': wt, 'mut': wt, 'labels': 0.0, 'cluster': torch.tensor(-1, dtype=torch.long)}
