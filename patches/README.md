# Patch layout

Only the mandatory clean-tree patch chain is shipped.

- `gguf-plugin/`: apply to `vllm-project/vllm`'s GGUF plugin at the commit in
  `pins.env`.
- `vllm/`: apply to vLLM at the commit in `pins.env`, in filename order.
- `optional/`: quarantined Hopper FlashMLA experiment; never apply during the
  first DSpark correctness run.

Superseded patches 0002/0003 and their 0006 reversion are deliberately absent.
The graph patch is published as `0014` to avoid colliding with the DSpark
`0010`; its patch payload and hash are unchanged from its validated source.

Use `scripts/apply_patches.sh`, not a shell glob.
