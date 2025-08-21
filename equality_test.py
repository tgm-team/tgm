import torch

TGM_PREDS, DYG_PREDS = './tgm_out.txt', '../DyGLib_TGB/dyg_out.txt'


def load_batches(filename):
    batches, current_batch, current_desc = {}, None, None
    with open(filename) as f:
        for line in map(str.strip, f):
            if not line:
                continue
            if line.startswith('BATCH'):
                _, b, d = line.split()
                current_batch, current_desc = int(b), d
                batches.setdefault(current_batch, {})[d] = []
            else:
                floats = list(map(float, line.split()))
                batches[current_batch][current_desc].extend(floats)
    for b in batches:
        for d in batches[b]:
            batches[b][d] = torch.tensor(batches[b][d], dtype=torch.float32)
    return batches


def compare_batches(file_a, file_b, atol=1e-7, rtol=1e-6):
    a, b = load_batches(file_a), load_batches(file_b)
    for batch_idx in sorted(a.keys()):
        for desc in a[batch_idx].keys():
            if desc not in b[batch_idx]:
                print(f"⚠️ Missing tensor '{desc}' in file B at batch {batch_idx}")
                continue
            try:
                torch.testing.assert_close(
                    a[batch_idx][desc], b[batch_idx][desc], rtol=rtol, atol=atol
                )
            except AssertionError:
                diff = (a[batch_idx][desc] - b[batch_idx][desc]).abs()
                idx, max_diff = torch.argmax(diff).item(), diff.max().item()
                print(f"❌ Divergence in batch {batch_idx}, tensor '{desc}'")
                print(f'   Max diff: {max_diff:.6g} at index {idx}')
                print(
                    f'   A={a[batch_idx][desc][idx].item()}, B={b[batch_idx][desc][idx].item()}'
                )
                return
    print('✅ All batches/tensors match within tolerance')


compare_batches(TGM_PREDS, DYG_PREDS)
