import os
import shutil
import subprocess
import tempfile

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()


class CITask(BaseModel):
    commit: str
    branch: str
    pr_number: str


CLUSTER_TOKEN = os.getenv('CLUSTER_TOKEN')
REPO_URL = os.getenv('REPO_URL')


@app.post('/')
async def trigger_ci(task: CITask, authorization: str = Header(None)):
    if authorization != f'Bearer {CLUSTER_TOKEN}':
        raise HTTPException(status_code=401, detail='Unauthorized')

    temp_dir = tempfile.mkdtemp(prefix='ci_run_')
    try:
        subprocess.run(['git', 'clone', REPO_URL, temp_dir], check=True)
        subprocess.run(['git', 'checkout', task.commit], cwd=temp_dir, check=True)

        result = subprocess.run(
            ['bash', './run_ci_tests.sh'],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {'status': 'failed', 'output': result.stdout + result.stderr}

        return {'status': 'success', 'output': result.stdout}

    except Exception as e:
        return {'status': 'error', 'detail': str(e)}
    finally:
        shutil.rmtree(temp_dir)
