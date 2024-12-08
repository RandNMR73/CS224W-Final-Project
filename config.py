import torch 

ROOT_DIR = "./data"
NEIGHBORS_PER_NODE = 128
DEPTH = 2
NUM_WORKERS = 0
BATCH_SIZE = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Dictionary that maps RelBench database to their associated tasks
RELBENCH_DATASETS = {
    "rel-amazon": ["user-churn", "item-churn", "user-ltv", "item-ltv", "user-item-purchase", "user-item-rate", "user-item-review"], 
    "rel-stack": ["user-engagement", "user-badge", "post-votes", "user-post-comment", "post-post-related"],
    "rel-trial": ["study-outcome", "study-adverse", "site-success", "condition-sponsor-run", "site-sponsor-run"],
    "rel-f1": ["driver-dnf", "driver-top3", "driver-position"], 
    "rel-hm": ["user-churn", "item-sales", "user-item-purchase"],
    "rel-event": ["user-repeat", "user-ignore", "user-attendance"],
    "rel-avito": ["user-visits", "user-clicks", "ad-ctr", "user-ad-visit"] 
}
