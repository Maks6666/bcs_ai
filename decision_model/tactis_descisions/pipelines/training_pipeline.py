from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import preprocess_data
from steps.tune_model import tune_model
from steps.train_model import train_model
from steps.eval_model import eval

@pipeline(enable_cache=False)
def run_pipeline(data_link, n_trails, model_name):
    data = ingest_data(data_link=data_link)
    x_train, x_test, y_train, y_test = preprocess_data(data=data)
    params, model_name = tune_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, n_trials=n_trails, model_name=model_name)
    trained_model = train_model(x_train=x_train, y_train=y_train, model_name=model_name, params=params)
    f1_score, accuracy_score, rmse_score = eval(model=trained_model, x_test=x_test, y_test=y_test)
    