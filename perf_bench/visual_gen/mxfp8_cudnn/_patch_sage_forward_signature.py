"""Make PR #13570's TrtllmAttention.forward() compatible with wheel callers
that don't pass batch_size / seq_len. Derive them from q.shape when missing.
"""

p = "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/visual_gen/attention_backend/trtllm.py"
s = open(p).read()

old_sig = """    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        seq_len_kv: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:"""

new_sig = """    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        seq_len_kv: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:"""

if old_sig in s:
    s = s.replace(old_sig, new_sig)
    print("forward() signature relaxed")
else:
    print("forward() signature already relaxed")

# Inside forward(), derive batch_size and seq_len from q.shape when None.
old_kv = """        kv_seq_len = seq_len_kv if seq_len_kv is not None else seq_len
        prepared_metadata = self._prepare_metadata(batch_size, seq_len)"""

new_kv = """        # Derive defaults from q shape so callers that don't pass them keep working
        # (wheel's modules/attention.py only forwards q/k/v/**kwargs).
        if batch_size is None:
            batch_size = q.shape[0]
        if seq_len is None:
            seq_len = q.shape[1]
        kv_seq_len = seq_len_kv if seq_len_kv is not None else (
            k.shape[1] if k is not None and k.dim() == 4 else seq_len
        )
        prepared_metadata = self._prepare_metadata(batch_size, seq_len)"""

if old_kv in s:
    s = s.replace(old_kv, new_kv)
    print("kv-derivation patched")
else:
    print("kv-derivation already patched")

open(p, "w").write(s)
