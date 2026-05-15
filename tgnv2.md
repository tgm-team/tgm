# TGNv2 Port Plan

## Summary

- Add TGNv2 as reusable TGM modules instead of keeping the implementation example-local.
- Keep the existing TGN APIs backward compatible.
- Draft runnable TGNv2 examples for link prediction and node property prediction using TGM data loaders, hooks, logging, and decoders.

## Key Changes

- Add `EncodeIndexMessage` to concatenate source memory, destination memory, raw edge message, source ID encoding, destination ID encoding, and time encoding.
- Add `TGNv2Memory` as a separate memory module that mirrors `TGNMemory` but passes trainable source and destination node-ID encodings into the message module.
- Export the new classes from `tgm.nn.encoder` and `tgm.nn`.
- Implement `examples/linkproppred/tgnv2.py` from the existing TGM TGN link-prediction example, replacing `TGNMemory + IdentityMessage` with `TGNv2Memory + EncodeIndexMessage`.
- Add `examples/nodeproppred/tgnv2.py` using the same reusable modules for node-property parity.

## Test Plan

- Extend TGN unit tests with `EncodeIndexMessage` shape/content checks.
- Add `TGNv2Memory` train/eval and `update_state` smoke tests.
- Add an integration smoke test for the link prediction example on `tgbl-wiki`.
- Run focused verification with `pytest test/unit/test_nn/test_tgn.py`.

## Assumptions

- The upstream behavior is implemented in TGM style rather than copied verbatim.
- `index_dim` defaults to `memory_dim` in examples when not explicitly provided.
- Existing TGN tests and examples should continue to work unchanged.
