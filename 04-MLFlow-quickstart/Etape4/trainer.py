from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient

from utils import compute_rmse
from data import get_data, clean_data
from encoders import DistanceTransformer, TimeFeaturesEncoder

MLFLOW_URI = "local_host"

class Trainer:
    def __init__(self, experiment_name):
        self.data = clean_data(get_data())
        self.experiment_name = experiment_name

    
    def set_pipeline(self):
        dist_pipe = Pipeline([("dist_trans", DistanceTransformer()), ("std_scale", StandardScaler())])
        time_pipe = Pipeline([("time_encode", TimeFeaturesEncoder()), ("one_hot", OneHotEncoder(handle_unknown="ignore"))])
        preproc_pipe = ColumnTransformer(transformers=[("distance", dist_pipe, 
                                                        ["pickup_latitude", 
                                                        "pickup_longitude", 
                                                        "dropoff_latitude", 
                                                        "dropoff_longitude"]),
                                                    ("time", time_pipe,
                                                        ["pickup_datetime"])])
        pipe = Pipeline([("preproc", preproc_pipe), ("model", LinearRegression())])
        return pipe
    
    def run(self, X_train, y_train, pipeline):
        pipeline.fit(X_train, y_train)
        return pipeline

    def evaluate(self, X_test, y_test, pipeline):
        y_pred = pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)
    
    @memoized_property
    def mlflow_client(self):
        #mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name)
    
    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)
    
    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":
    trainer = Trainer("test_trainer")
    y = trainer.data.fare_amount
    X = trainer.data.drop("fare_amount", axis=1)
    pipe = trainer.set_pipeline()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    trainer.run(X_train, y_train, pipe)
    rmse = trainer.evaluate(X_val, y_val, pipe)
    trainer.mlflow_log_param("model", "LinearRegression")
    trainer.mlflow_log_metric("rmse", rmse)
    print(rmse)
