# install packages
pip install --quiet "torch-geometric ==2.1.*" "torch-sparse ==0.6.*" "torch-scatter ==2.1.*" "torchmetrics >=1.0,<1.8" "torch ==2.1.2" "torchvision" "pytorch-lightning >=2.0,<2.6" "matplotlib" "seaborn" "torch >=1.8.1,<2.8" "numpy <2.0" "torch-cluster ==1.6.*" "torch-spline-conv ==1.2.*" "numpy <3.0"

# load
import autogluon
import torch
import pandas
import numpy
import autogl

from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) AutoGL will download & load the dataset for you
dataset = build_dataset_from_name("cora")  # Cora Planetoid dataset
# AutoGL’s dataset builder wraps PyG datasets and handles splits/featurization internally.  # noqa

# 2) Build an AutoML solver:
solver = AutoNodeClassifier(
    feature_module="deepgl",            # automatic feature engineering
    graph_models=["gcn", "gat", "sage"],# candidate backbones (GCN/GAT/SAGE)
    hpo_module="anneal",                # hyperparameter optimizer
    ensemble_module="voting",           # automatic ensembling
    device=device
)

# 3) Optimize within a time budget (e.g., 15 minutes)
solver.fit(dataset, time_limit=900)     # seconds

# 4) Inspect leaderboard and evaluate on test split
print(solver.get_leaderboard().show())
metrics = solver.evaluate(metric="acc") # test accuracy
print("Test accuracy:", metrics)

