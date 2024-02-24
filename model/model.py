import torch
import torch.nn as nn
from .obsEncoder import ObsEncoder
from .target_prediction import TargetPred
from .utils import CrossAttention, MLP,DecoderResCat,masked_softmax
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        dropout = cfg["dropout"]
        d_model = cfg['d_model']
        causal_num_layers=cfg['causal_num_layers']
        self.nhead = cfg['head_num']
        self.future_num_frames = cfg['future_num_frames']
        self.act_dim = cfg['act_dim']
        self.causal_len = cfg['causal_len']
        self.causal_interval = cfg['causal_interval']
        attention_mask = torch.ones((self.causal_len, self.causal_len)).bool()
        causal_mask = ~torch.tril(attention_mask, diagonal=0)  # diagonal=0, keep the diagonal
        self.register_buffer("causal_mask", causal_mask)
        self.embed_timestep = nn.Embedding(self.causal_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=d_model,
            nhead=self.nhead,
            dropout=dropout,
            batch_first=True)
        self.embed_state = ObsEncoder(cfg,encoder_layer)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=causal_num_layers)
        self.predict_action=nn.Linear(d_model,   self.act_dim *  self.future_num_frames)
        self.target_pred_layer = TargetPred(
            in_channels=d_model,
            hidden_dim=d_model,
            m=50,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.cross_attention=CrossAttention(hidden_size=d_model)
        self.candidate_encoder = nn.Sequential(
                MLP(3, d_model),
                MLP(hidden_size=d_model),
                MLP(hidden_size=d_model))
        self.goals_2D_decoder = DecoderResCat(hidden_size=d_model, in_features=d_model * 2, out_features=1)

    def forward(self,batch):

        state_embeddings,ego_mask=self.embed_state(batch)
        device=state_embeddings.device
        target_candidate=batch["target_candidates"]
        #target_candidate=target_candidate.view(1,-1,3)
        mask=batch["mask"]
        batch_size, n, _ = target_candidate.size()
        #goals_2D_hidden_attention = self.cross_attention(goals_2D_hidden.unsqueeze(0), inputs[i][:inputs_lengths[i]].unsqueeze(0)).squeeze(0)
        candidate_embeddings=self.candidate_encoder(target_candidate.float())
        goals_2D_hidden_attention = self.cross_attention(candidate_embeddings, state_embeddings)
        scores=self.goals_2D_decoder(torch.cat([candidate_embeddings,goals_2D_hidden_attention],dim=-1)).squeeze(-1)
        scores=masked_softmax(scores,mask.squeeze(2),dim=-1)
        value,indices=scores.topk(100, dim=1)
        ad_value=F.softmax(value, dim=-1)
        prob = torch.zeros((batch_size,n)).to(device)
        for batch in range(batch_size):
            for dim in range(100):
                prob[batch][indices[batch][dim]]=ad_value[batch][dim]
        target_prob, offset = self.target_pred_layer(state_embeddings, target_candidate, mask)

        return {"target_prob":scores,
                "offset":offset }
    
    def get_masked_score(self,target_candidate,state_embeddings,mask):
        candidate_embeddings=self.candidate_encoder(target_candidate.float())
        goals_2D_hidden_attention = self.cross_attention(candidate_embeddings, state_embeddings)
        scores=self.goals_2D_decoder(torch.cat([candidate_embeddings,goals_2D_hidden_attention],dim=-1)).squeeze(-1)
        scores=masked_softmax(scores,mask.squeeze(2),dim=-1)
        return scores

    def get_score(self,target_candidate,state_embeddings):
        candidate_embeddings=self.candidate_encoder(target_candidate.float())
        goals_2D_hidden_attention = self.cross_attention(candidate_embeddings, state_embeddings)
        scores=self.goals_2D_decoder(torch.cat([candidate_embeddings,goals_2D_hidden_attention],dim=-1)).squeeze(-1)
        scores=F.softmax(scores,dim=-1)
        return scores

    
