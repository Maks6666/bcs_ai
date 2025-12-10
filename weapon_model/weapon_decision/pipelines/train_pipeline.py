from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_tune import tune
from steps.model_train import train_model
from steps.model_eval import eval

@pipeline
def run_pipeline(link: str, model_name: str, n_trials: int):
    data = ingest_data(link)
    x_train, x_test, y_train, y_test = clean_data(data)
    best_params = tune(model_name, n_trials, x_train, x_test, y_train, y_test)
    model = train_model(x_train, y_train, best_params, model_name)
    accuracy_score, f1, rmse = eval(model, x_test, y_test)
