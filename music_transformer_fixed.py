import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicTransformer(nn.Module):
    """
    Fixed Music Transformer based on LakhNES architecture
    
    This implementation includes:
    - Transformer-XL style memory mechanism
    - Simplified positional encoding (added to embeddings, not attention)
    - Causal attention masking
    """
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_head=8,
        d_head=64,
        d_inner=2048,
        n_layer=12,
        dropout=0.1,
        dropatt=0.0,
        pre_lnorm=False,
        tgt_len=512,
        ext_len=0,
        mem_len=512,
        tie_weight=True,
        clamp_len=-1,
    ):
        super(MusicTransformer, self).__init__()
        
        self.n_token = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        
        # Token embedding
        self.word_emb = nn.Embedding(vocab_size, d_model)
        
        self.drop = nn.Dropout(dropout)
        
        self.n_layer = n_layer
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len
        
        # Positional encoding
        self.pos_emb = nn.Parameter(torch.zeros(self.max_klen, d_model))
        nn.init.normal_(self.pos_emb, 0.0, 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                TransformerLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm
                )
            )
        
        # Output projection
        self.out_layer = nn.Linear(d_model, vocab_size)
        if tie_weight:
            self.out_layer.weight = self.word_emb.weight
        
        self.clamp_len = clamp_len
        
        # Initialize parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def init_mems(self, device=None):
        if self.mem_len > 0:
            mems = []
            if device is None:
                device = next(self.parameters()).device
            
            for i in range(self.n_layer):
                empty = torch.empty(0, dtype=torch.float, device=device)
                mems.append(empty)
            return mems
        else:
            return None
    
    def _update_mems(self, hids, mems, qlen, mlen):
        if mems is None:
            return None
        
        assert len(hids) == len(mems), 'len(hids) != len(mems)'
        
        with torch.no_grad():
            new_mems = []
            # Cache up to mem_len tokens, skipping the most recent `ext_len`
            end_idx = mlen + max(0, qlen - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        
        return new_mems
    
    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()
        
        # Token embeddings
        word_emb = self.word_emb(dec_inp)
        
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        
        # Positional embeddings - simplified approach
        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                               dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb[:klen]
        
        # Add positional embeddings to word embeddings
        pos_emb_qlen = pos_emb[-qlen:].unsqueeze(1)  # (qlen, 1, d_model)
        core_out = self.drop(word_emb + pos_emb_qlen)
        
        hids = []
        # Create causal attention mask
        dec_attn_mask = torch.triu(
            torch.ones(qlen, klen, device=word_emb.device),
            diagonal=1+mlen).bool()
        
        # Run through transformer layers
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, dec_attn_mask, mems=mems_i)
            hids.append(core_out)
        
        core_out = self.drop(core_out)
        
        # Update memory
        new_mems = self._update_mems(hids, mems, qlen, mlen)
        
        return core_out, new_mems
    
    def forward(self, data, target=None, mems=None):
        # data: (batch_size, seq_len) - we'll transpose it
        data = data.transpose(0, 1).contiguous()  # (seq_len, batch_size)
        
        if mems is None:
            mems = self.init_mems(device=data.device)
        
        hidden, new_mems = self._forward(data, mems=mems)
        
        pred_hid = hidden
        output = self.out_layer(pred_hid)
        output = output.transpose(0, 1).contiguous()
        
        print("Input shape:", data.shape)
        print("Output shape:", output.shape)
        print("Number of memory states:", len(mems) if mems else 0)
        
        return output, new_mems


class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 dropatt=0.0, pre_lnorm=False):
        super(TransformerLayer, self).__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.d_inner = d_inner
        self.dropout = dropout
        
        # Separate linear layers for q, k, v
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)
        
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        
        self.pos_ff = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        self.pre_lnorm = pre_lnorm
        
    def forward(self, h, attn_mask=None, mems=None):
        # Multihead attention
        if self.pre_lnorm:
            h = self.layer_norm1(h)
        
        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h
        
        qlen = h.size(0)
        klen = c.size(0)
        
        # Compute q from current input
        q = self.q_net(h)
        
        # Compute k, v from concatenated sequence (includes memory)
        kv = self.kv_net(c)
        k, v = torch.chunk(kv, 2, dim=-1)
        k = k.view(klen, -1, self.n_head, self.d_head)
        k = k.permute(2, 1, 0, 3)
        v = v.view(klen, -1, self.n_head, self.d_head)
        v = v.permute(2, 1, 0, 3)
        
        # --- Apply rotary positional embeddings to q and k ---
        sin, cos = _get_sin_cos(
            klen, self.d_head, device=q.device, dtype=q.dtype
        )
        sin = sin.unsqueeze(0).unsqueeze(0)  # 1,1,klen,d_head
        cos = cos.unsqueeze(0).unsqueeze(0)

        q = _apply_rotary(q, sin[:, :, -qlen:, :], cos[:, :, -qlen:, :])
        k = _apply_rotary(k, sin, cos)
        
        # Attention scores
        attn_score = torch.matmul(q, k.transpose(-1, -2))
        attn_score.mul_(1 / (self.d_head ** 0.5))
        
        # Attention mask
        if attn_mask is not None and attn_mask.any():
            attn_score = attn_score.float().masked_fill(
                attn_mask[None, None, :, :], -float('inf')).type_as(attn_score)
        
        # Attention weights
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropatt(attn_prob)
        
        # Attention output
        attn_vec = torch.matmul(attn_prob, v)
        attn_vec = attn_vec.transpose(1, 2).contiguous()
        attn_vec = attn_vec.view(qlen, -1, self.n_head * self.d_head)
        
        # Linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        
        if self.pre_lnorm:
            output = h + attn_out
        else:
            output = self.layer_norm1(h + attn_out)
        
        # Position-wise feedforward
        if self.pre_lnorm:
            output = self.layer_norm2(output)
        
        output2 = self.pos_ff(output)
        
        if self.pre_lnorm:
            output = output + output2
        else:
            output = self.layer_norm2(output + output2)
        
        return output


# --- Rotary positional embedding helpers (relative positional information) ---


def _get_sin_cos(seq_len: int, d_head: int, device, dtype):
    """Return sinusoid sin and cos matrices used for rotary embeddings.

    Both tensors have shape (seq_len, d_head) and are ready for broadcasting.
    They will later be unsqueezed to 1, 1, seq_len, d_head before use.
    """
    arange = torch.arange(0, d_head, 2, device=device, dtype=dtype)
    inv_freq = 1.0 / (10000 ** (arange / d_head))
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    sinusoid_inp = torch.einsum('i,j->ij', positions, inv_freq)
    sin = torch.repeat_interleave(torch.sin(sinusoid_inp), 2, dim=-1)
    cos = torch.repeat_interleave(torch.cos(sinusoid_inp), 2, dim=-1)
    return sin, cos


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper that swaps and negates every other dimension (used by RoPE)."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(-2)


def _apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to x (… d_head)."""
    # (x * cos) + (rotate_half(x) * sin)
    rotated_component = _rotate_half(x) * sin
    return (x * cos) + rotated_component


# Example usage and testing
if __name__ == "__main__":
    # Test the model
    vocab_size = 1127  # From your notebook
    batch_size = 4
    seq_len = 128
    
    # Create model with LakhNES-like configuration
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=512,
        n_head=8,
        d_head=64,
        d_inner=2048,
        n_layer=12,
        dropout=0.1,
        tgt_len=512,
        mem_len=512
    )
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output, mems = model(input_ids)
    
    print("Input shape:", input_ids.shape)
    print("Output shape:", output.shape)
    print("Number of memory states:", len(mems) if mems else 0) 