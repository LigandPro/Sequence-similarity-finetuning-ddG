import torch
import torch.nn.functional as F
from .base_blocks import BaseddGRegressor, MLP

class ddGRegressor(BaseddGRegressor):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
    
    def _init_model_components(self, cfg):
        """Initialize model components including the context embedder"""
        
        
        self.precoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.embeding_dim, 
                nhead=cfg.MODEL.NUM_HEADS, 
                batch_first=True, 
                dropout=cfg.TRAIN.DROPOUT, 
                norm_first=True, 
                activation="gelu", 
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
                activation="gelu", 
                dim_feedforward=cfg.MODEL.HIDDEN_SIZE*cfg.MODEL.SELFATT_DIM_EXPANSION
            ),
            num_layers=cfg.MODEL.DECODER_LAYERS
        )
        self.model_modules.append(('DEC', self.decoder))
        
        self.mix_layer = torch.nn.Sequential(
            torch.nn.Linear(self.embeding_dim, self.embeding_dim*cfg.MODEL.SELFATT_DIM_EXPANSION),
            torch.nn.LayerNorm(self.embeding_dim*cfg.MODEL.SELFATT_DIM_EXPANSION),
            torch.nn.GELU(),
            torch.nn.Dropout(cfg.TRAIN.DROPOUT),
            torch.nn.Linear(self.embeding_dim*cfg.MODEL.SELFATT_DIM_EXPANSION, self.embeding_dim),
            torch.nn.Dropout(cfg.TRAIN.DROPOUT)
        )
        self.model_modules.append(('MIX', self.mix_layer))
        
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.embeding_dim, 
                nhead=cfg.MODEL.NUM_HEADS, 
                batch_first=True, 
                dropout=cfg.TRAIN.DROPOUT,
                norm_first=True, 
                activation="gelu", 
                dim_feedforward=cfg.MODEL.HIDDEN_SIZE*cfg.MODEL.SELFATT_DIM_EXPANSION
            ),
            num_layers=cfg.MODEL.ENCODER_LAYERS
        )
        self.model_modules.append(('ENC', self.encoder))
        
        self.head = MLP(cfg, embedding_dim_multiplier=1)
        self.model_modules.append(('H', self.head))
        
    def _preprocess(self, wt, mut, wt_att_mask, mut_att_mask):
        """Process the initial embeddings - can be overridden by subclasses."""
        wt, mut = self.precoder(
            src=wt, 
            src_key_padding_mask=wt_att_mask
        ), self.precoder(
            src=mut, 
            src_key_padding_mask=mut_att_mask
        )
        return wt, mut

    def _mix(self, wt, mut, wt_att_mask, mut_att_mask):
        """Apply recycling steps - can be overridden by subclasses."""
        
        mut_w_context = self.decoder(
            tgt=mut, 
            memory=mut-wt, 
            tgt_key_padding_mask=mut_att_mask, 
            memory_key_padding_mask=wt_att_mask
        )
        wt_w_context = self.decoder(
            tgt=wt, 
            memory=wt-mut, 
            tgt_key_padding_mask=wt_att_mask, 
            memory_key_padding_mask=mut_att_mask
        )

        return wt_w_context, mut_w_context
    
    def _aggregate(self, wt, mut, wt_att_mask):
        """Aggregate with both antisymmetric and symmetric components"""
        context = mut - wt
        context = self.mix_layer(context)
        return context

    def _postprocess(self, context, mask):

        context = self.encoder(src=context, src_key_padding_mask=mask)
        
        # Pool only pooling embedding
        context = context[:, 0, :]
        context = context.squeeze()
        return context




