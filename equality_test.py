TGM_PREDS = './tgm_out.txt'
DYG_PREDS = '../DyGLib_TGB/dyg_out.txt'


def load_batches(filename):
    batches = {}
    with open(filename) as f:
        current_batch = None
        for line in f:
            line = line.strip()
            if line.startswith('BATCH'):
                current_batch = int(line.split()[1])
                batches[current_batch] = []
            else:
                preds = list(map(int, line.split()))
                batches[current_batch].extend(preds)
    return batches


tgm = load_batches(TGM_PREDS)
dyg = load_batches(DYG_PREDS)

for i in sorted(tgm.keys()):
    if tgm[i] != dyg[i]:
        diffs = [j for j, (x, y) in enumerate(zip(tgm[i], dyg[i])) if x != y]
        print(
            f'❌ Divergence in batch {i}, {len(diffs)} mismatches (first 10: {diffs[:10]})'
        )
        break
else:
    print('✅ All batches match exactly')
