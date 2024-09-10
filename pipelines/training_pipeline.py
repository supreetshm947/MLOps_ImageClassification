from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model


@pipeline(enable_cache=False)
def train_pipeline(data_path):
    train_loader, test_loader, num_classes = ingest_data(data_path)
    # model, criterion =
    # train_model(train_loader, num_classes)
    # evaluate_model(model, criterion, test_loader)
    # print(num_classes)
    model, criterion = train_model(train_loader, num_classes)
    evaluate_model(model, criterion, test_loader)