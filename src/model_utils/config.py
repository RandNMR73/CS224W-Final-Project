import torch 
import relbench

# Directory and resource configurations
ROOT_DIR = "./data"
NEIGHBORS_PER_NODE = 128
DEPTH = 2
NUM_WORKERS = 0
BATCH_SIZE = 512

# Model-related parameters
EPOCHS = 30
EPOCHS_TO_SAVE = 2
EPOCHS_TO_SWITCH = 50

# LRS for rel-f1
LRS = {"driver-dnf": 0.005,
       "driver-position" : 0.005,
       "driver-top3": 0.005
       }
# LRS for rel-trial
# LRS = {
#     "study-outcome": 0.005,
#     "study-adverse": 0.005,
#     "site-success": 0.005,
#     "condition-sponsor-run": 0.005,
#     "site-sponsor-run": 0.005,
#     "user-visits": 0.005
# }

# Regularization parameters
WEIGHT_DECAY = 0.01
DROPOUT = 0.1

# Model architecture parameters
EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 5

# Checkpointing configurations
CHECKPOINT = True
CHECKPOINT_FOLDER = "./checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RelBench datasets mapping
RELBENCH_DATASETS = {
    "rel-amazon": ["user-churn", "item-churn", "user-ltv", "item-ltv", "user-item-purchase", "user-item-rate", "user-item-review"], 
    "rel-stack": ["user-engagement", "user-badge", "post-votes", "user-post-comment", "post-post-related"],
    "rel-trial": ["study-outcome", "study-adverse", "site-success", "condition-sponsor-run", "site-sponsor-run"],
    "rel-f1": ["driver-dnf", "driver-top3", "driver-position"], 
    "rel-hm": ["user-churn", "item-sales", "user-item-purchase"],
    "rel-event": ["user-repeat", "user-ignore", "user-attendance"],
    "rel-avito": ["user-visits", "user-clicks", "ad-ctr", "user-ad-visit"] 
}

# Configuration for metrics evaluation
HIGHER_IS_BETTER = {
    relbench.metrics.accuracy: True,
    relbench.metrics.log_loss: False,
    relbench.metrics.f1: True,
    relbench.metrics.roc_auc: True,
    relbench.metrics.average_precision: True,
    relbench.metrics.auprc: True,
    relbench.metrics.macro_f1: True,
    relbench.metrics.micro_f1: True,
    relbench.metrics.mae: False,
    relbench.metrics.mse: False,
    relbench.metrics.rmse: False,
    relbench.metrics.r2: True,
    relbench.metrics.multilabel_auprc_micro: True,
    relbench.metrics.multilabel_auprc_macro: True,
    relbench.metrics.multilabel_auroc_micro: True,
    relbench.metrics.multilabel_auroc_macro: True,
    relbench.metrics.multilabel_f1_micro: True,
    relbench.metrics.multilabel_f1_macro: True,
    relbench.metrics.multilabel_recall_micro: True,
    relbench.metrics.multilabel_recall_macro: True,
    relbench.metrics.multilabel_precision_micro: True,
    relbench.metrics.multilabel_precision_macro: True,
    relbench.metrics.link_prediction_recall: True,
    relbench.metrics.link_prediction_precision: True,
    relbench.metrics.link_prediction_map: True,
}
