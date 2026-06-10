# import mlflow
# from sqlalchemy import create_engine, text

# mlflow.set_tracking_uri('sqlite:///./mlruns.db')
# client = mlflow.tracking.MlflowClient()

# for e in client.search_experiments(view_type='ALL'):
#     if 'fall-detection' in e.name:
#         client.delete_experiment(e.experiment_id)
#         print(f'Soft-deleted: {e.name} (id={e.experiment_id})')

# engine = create_engine('sqlite:///./mlruns.db')
# with engine.connect() as conn:
#     conn.execute(text('DELETE FROM runs WHERE experiment_id != 0'))
#     conn.execute(text('DELETE FROM experiments WHERE experiment_id != 0'))
#     conn.commit()
# print('Permanently deleted all non-default experiments.')


################################################################################
# # Check which version number was just created:
# import mlflow

# mlflow.set_tracking_uri('sqlite:///./mlruns.db')
# client = mlflow.tracking.MlflowClient()
# versions = client.search_model_versions("name='fall-detection-xgboost'")
# for v in versions:
#     print(f'Version {v.version}  run_id={v.run_id}  status={v.status}')


################################################################################
# The new version will have the run_id that matches your latest retrain. Then promote it:
import mlflow

# mlflow.set_tracking_uri('sqlite:///./mlruns.db') # use this only when the default local file-based store is used
mlflow.set_tracking_uri('http://localhost:5000') # use this when connected to a remote/dockernized MLflow server (e.g. Docker container with Postgres backend)
client = mlflow.tracking.MlflowClient()
# Replace X with the version number from the output above
client.set_registered_model_alias('fall-detection-xgboost', 'Production', 1)
print('Done')