import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import EsmModel
from transformers import AutoModelForMaskedLM
from abc import ABC, abstractmethod

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate, bias=True, activation='silu'):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.c_fc    = nn.Linear(in_dim, out_dim, bias=bias)
        
        # Select activation function
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1)
        elif activation == 'linear':
            self.act = nn.Identity()
            
        self.dropout = nn.Dropout(dropout_rate)
        self.mlpf = lambda x: self.dropout(self.act(self.c_fc(self.ln(x))))

    def forward(self, x):
        x = self.mlpf(x)
        return x

class MLP(nn.Module):
    # Iteratively decreasing hidden size MLPBlocks, final out_dim = 1
    def __init__(self, cfg, embedding_dim_multiplier=2, input_dim=None):
        super().__init__()
        self.cfg = cfg
        if not input_dim:
            input_dim = cfg.MODEL.HIDDEN_SIZE
        num_layers = cfg.MODEL.MLP_LAYERS
        hidden_sizes = (embedding_dim_multiplier * input_dim * (1 - np.linspace(0, 1, num_layers+1))).astype(int)
        hidden_sizes[-1] = 1
        
        self.mlp_blocks = nn.ModuleList([
            MLPBlock(
                hidden_sizes[i], 
                hidden_sizes[i+1], 
                cfg.TRAIN.DROPOUT,
                activation='silu' if i < num_layers-1 else 'linear'  # Last layer uses linear
            ) for i in range(num_layers)
        ])

    def forward(self, x):
        for block in self.mlp_blocks:
            x = block(x)
        return x


class BaseddGRegressor(torch.nn.Module, ABC):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        
        self.esm = EsmModel.from_pretrained(cfg.MODEL.ESM_CHECKPOINT, add_pooling_layer=False)
        
        cfg.defrost()
        cfg.MODEL.HIDDEN_SIZE = self.esm.config.hidden_size
        cfg.freeze()        
        
        
        self.esm.requires_grad_(False)
        self.esm_config = self.esm.config
        self.embeding_dim = self.esm_config.hidden_size
        self.esm.eval()
        
        self.model_modules = []
        self.main_input_name = "clusters"

        self._init_model_components(cfg)
        
        self.loss_fn = self._init_loss_functions(cfg)
        
        self.cfg = cfg
    
    def forward(self, labels, **inputs):
        encodings = inputs["encodings"]
        wt, mut, wt_att_mask, mut_att_mask = self._esm_embeddings(encodings)
        
        wt, mut = self._preprocess(wt, mut, wt_att_mask, mut_att_mask)
        
        wt, mut = self._mix(wt, mut, wt_att_mask, mut_att_mask)
        
        context = self._aggregate(wt, mut, wt_att_mask)
        
        context = self._postprocess(context, wt_att_mask) 
        
        logits = self.head(context)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            prediction_loss = self.loss_fn(logits, labels)
            loss = prediction_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            #inputs=inputs,
            hidden_states=None,
            attentions=None,
        )
    
    def _esm_embeddings(self, inputs):
        with torch.no_grad():
            wt, mut = inputs["wt"], inputs["mut"]
            wt_att_mask, mut_att_mask = wt["attention_mask"].clone().to(torch.bool), mut["attention_mask"].clone().to(torch.bool)
            wt_att_mask, mut_att_mask = ~wt_att_mask, ~mut_att_mask # True to skip as torch.MultiheadAttention expects
            wt, mut = self.esm(**wt).last_hidden_state, self.esm(**mut).last_hidden_state

            # Ensure wt and mut have no gradients turned on
            wt = wt.detach()
            mut = mut.detach()

        return wt, mut, wt_att_mask, mut_att_mask
    
    def _init_model_components(self, cfg):
        """Initialize core model components - can be overridden by subclasses.
        Uses 4 main components: precoder, decoder, encoder, head"""
        
        self.precoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=self.embeding_dim, 
                nhead=cfg.MODEL.NUM_HEADS, 
                batch_first=True, 
                dropout=cfg.TRAIN.DROPOUT, 
                norm_first=True,
                activation="silu", 
                dim_feedforward=cfg.MODEL.HIDDEN_SIZE*cfg.MODEL.SELFATT_DIM_EXPANSION
            ), 
            num_layers=cfg.MODEL.PRECODER_LAYERS
        )
        self.model_modules.append(('PRE', self.precoder))
        
        self.decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=self.embeding_dim, 
                nhead=cfg.MODEL.NUM_HEADS, 
                batch_first=True, 
                dropout=cfg.TRAIN.DROPOUT, 
                norm_first=True, 
                activation="silu", 
                dim_feedforward=cfg.MODEL.HIDDEN_SIZE*cfg.MODEL.SELFATT_DIM_EXPANSION
            ),
            num_layers=cfg.MODEL.DECODER_LAYERS
        )
        self.model_modules.append(('DEC', self.decoder))
        
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.embeding_dim*2, 
                nhead=cfg.MODEL.NUM_HEADS, 
                batch_first=True, 
                dropout=cfg.TRAIN.DROPOUT, 
                norm_first=True, 
                activation="silu", 
                dim_feedforward=cfg.MODEL.HIDDEN_SIZE*cfg.MODEL.SELFATT_DIM_EXPANSION
            ),
            num_layers=cfg.MODEL.ENCODER_LAYERS
        )
        self.model_modules.append(('ENC', self.encoder))
        
        # embedding_dim_multiplier should be in alignment with _aggregate
        self.head = MLP(cfg, embedding_dim_multiplier=2)
        self.model_modules.append(('H', self.head))
    
    @abstractmethod
    def _preprocess(self, wt, mut, wt_att_mask, mut_att_mask):
        """Process the initial embeddings - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _preprocess")
    
    @abstractmethod
    def _mix(self, wt, mut, wt_att_mask, mut_att_mask):
        """Apply mixing of information from mut and wt - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _mix")
    
    @abstractmethod
    def _aggregate(self, wt, mut, wt_att_mask):
        """Aggregate representations of wt and mut - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _aggregate")
    
    @abstractmethod
    def _postprocess(self, context, wt_att_mask):
        """Post-process the context - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _postprocess")

    def _init_loss_functions(self, cfg):
        """Initialize loss functions - can be overridden by subclasses."""
        self.loss_fn = self._compute_loss
        if cfg.TRAIN.ALPHA > 0:
            self.loss_fn = self._compute_scaled_loss
        return self.loss_fn
    
    def _compute_loss(self, logits, labels):
        return F.mse_loss(logits, labels[:, None])

    def _compute_scaled_loss(self, logits, labels):
        """Compute scaled MSE loss for large label predictions."""
        mse = F.mse_loss(logits.float(), labels[:, None].float(), reduction="none")
        mse = mse.squeeze()
        weighted_loss = ((1 + self.cfg.TRAIN.ALPHA * labels**2) * mse).mean()
        return weighted_loss

    def _calculate_num_parameters(self, obj=None):
        if obj:
            return sum(p.numel() for p in obj.parameters()) / 1e6
        else:
            return sum(p.numel() for p in self.parameters()) / 1e6

def move_batch_to_device(batch, device):
    """Move batch and nested structures to device"""
    moved_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved_batch[k] = v.to(device)
        elif isinstance(v, dict):
            # Handle nested dictionaries like encodings
            moved_batch[k] = {}
            for nested_k, nested_v in v.items():
                if isinstance(nested_v, dict):
                    # Handle double-nested dictionaries like encodings["wt"] and encodings["mut"]
                    moved_batch[k][nested_k] = {}
                    for double_nested_k, double_nested_v in nested_v.items():
                        if isinstance(double_nested_v, torch.Tensor):
                            moved_batch[k][nested_k][double_nested_k] = double_nested_v.to(device)
                        else:
                            moved_batch[k][nested_k][double_nested_k] = double_nested_v
                elif isinstance(nested_v, torch.Tensor):
                    moved_batch[k][nested_k] = nested_v.to(device)
                else:
                    moved_batch[k][nested_k] = nested_v
        else:
            moved_batch[k] = v
    return moved_batch

