import torch

TGM_PREDS = './tgm_out.txt'
DYG_PREDS = '../DyGLib_TGB/dyg_out.txt'


def load_batches(filename):
    batches = {}
    current_batch = None
    current_desc = None

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('BATCH'):
                parts = line.split()
                current_batch = int(parts[1])
                current_desc = parts[2]
                batches.setdefault(current_batch, {})[current_desc] = []
            else:
                floats = list(map(float, line.split()))
                batches[current_batch][current_desc].extend(floats)

    for b in batches:
        for d in batches[b]:
            batches[b][d] = torch.tensor(batches[b][d], dtype=torch.float32)
    return batches


def compare_batches(file_a, file_b, atol=1e-6, rtol=1e-6):
    a = load_batches(file_a)
    b = load_batches(file_b)

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
                max_diff = diff.max().item()
                idx = torch.argmax(diff).item()
                print(f"❌ Divergence in batch {batch_idx}, tensor '{desc}'")
                print(f'   Max diff: {max_diff:.6g} at index {idx}')
                print(
                    f'   A={a[batch_idx][desc][idx].item()}, '
                    f'B={b[batch_idx][desc][idx].item()}'
                )
                return
    print('✅ All batches/tensors match within tolerance')


compare_batches(TGM_PREDS, DYG_PREDS)
