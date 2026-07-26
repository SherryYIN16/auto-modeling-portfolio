from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import uuid
import os
import json
from modeling_manager import ModelingFactory

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

tasks = {}; task_logs = {}
UPLOAD_DIR = 'backend/data'
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, f'{uuid.uuid4()}_{file.filename}')
    with open(file_path, 'wb') as buffer:
        buffer.write(await file.read())
    df = pd.read_csv(file_path)
    return {'columns': df.columns.tolist(), 'path': file_path}

@app.post('/model/start')
async def start_model(target: str, path: str, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'started', 'progress': 0, 'step': '初始化'}
    task_logs[task_id] = []
    background_tasks.add_task(run_real_modeling, task_id, path, target)
    return {'task_id': task_id, 'status': 'started'}

def run_real_modeling(task_id, path, target):
    try:
        def log(msg):
            task_logs[task_id].append(msg)
            print(f'DEBUG: Saving log: {msg}')
        report = ModelingFactory.run_pipeline(path, target, task_id, tasks, logger=log)
        print(f"DEBUG: Final report structure: {json.dumps(report, ensure_ascii=False)}")
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['report'] = report
    except Exception as e:
        tasks[task_id]['status'] = f'error: {str(e)}'
        print(f'Modeling Error: {e}')

@app.get('/model/logs/{task_id}')
async def get_logs(task_id: str):
    return {'logs': task_logs.get(task_id, [])}

@app.get('/model/status/{task_id}')
async def get_status(task_id: str):
    return tasks.get(task_id, {'status': 'not_found'})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
