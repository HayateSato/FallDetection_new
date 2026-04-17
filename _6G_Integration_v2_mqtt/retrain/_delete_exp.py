import mlflow
from sqlalchemy import create_engine, text

mlflow.set_tracking_uri('sqlite:///./mlruns.db')
client = mlflow.tracking.MlflowClient()

for e in client.search_experiments(view_type='ALL'):
    if 'fall-detection' in e.name:
        client.delete_experiment(e.experiment_id)
        print(f'Soft-deleted: {e.name} (id={e.experiment_id})')

engine = create_engine('sqlite:///./mlruns.db')
with engine.connect() as conn:
    conn.execute(text('DELETE FROM runs WHERE experiment_id != 0'))
    conn.execute(text('DELETE FROM experiments WHERE experiment_id != 0'))
    conn.commit()
print('Permanently deleted all non-default experiments.')
