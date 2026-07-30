from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    tasks[task_id] = {'status': 'started', 'progress': 0, 'step': '初始化', 'path': path, 'target': target}
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

@app.get('/model/export_code/{task_id}')
async def export_code(task_id: str):
    task = tasks.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={'error': 'Task not found'})
    
    # 自动获取当前 main.py 所在的目录并拼接路径
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'pipeline_template.py')
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    code = template.replace('{{target}}', task.get('target', 'target'))
    code = code.replace('{{data_path}}', task.get('path', 'data.csv'))
    return {'code': code}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8001)
