import pytest


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbl-wiki'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=0:03:00',
    ]
)
def test_edgebank_linkprop_pred_unlimited_memory(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/edgebank.py" \
    --dataset {dataset}"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbl-wiki'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=0:03:00',
    ]
)
def test_edgebank_linkprop_pred_fixed_memory(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/edgebank.py" \
    --dataset {dataset} --memory-mode fixed"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['GoogleLocal'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=8G',
        '--time=0:15:00',
    ]
)
def test_edgebank_tgb_seq_unlimited_memory(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/tgb_seq/edgebank.py" \
    --dataset {dataset}"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['thgl-software'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=8G',
        '--time=0:15:00',
    ]
)
def test_edgebank_linkprop_pred_unlimited_memory_thgl(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/thgl/edgebank.py" \
    --dataset {dataset}"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['thgl-software'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=8G',
        '--time=0:15:00',
    ]
)
def test_edgebank_linkprop_pred_fixed_memory_thgl(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/thgl/edgebank.py" \
    --dataset {dataset} --memory-mode fixed"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tkgl-smallpedia'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=8G',
        '--time=1:15:00',
    ]
)
def test_edgebank_linkprop_pred_unlimited_memory_tkgl(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/tkgl/edgebank.py" \
    --dataset {dataset}"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tkgl-smallpedia'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=8G',
        '--time=1:15:00',
    ]
)
def test_edgebank_linkprop_pred_fixed_memory_tkgl(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/tkgl/edgebank.py" \
    --dataset {dataset} --memory-mode fixed"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'
