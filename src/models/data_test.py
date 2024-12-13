# def load_data(dataset, task, train_table, val_table, test_table):
# def load_dataset(dataset_name, task_name):

# dick = {}
# for database_name, tasks in config.RELBENCH_DATASETS.items():
#     for task in tasks:
#         dataset, task, train_table, val_table, test_table = load_dataset(database_name, task)
#         loader_dict, col_stats_dict, entity_table = load_data(dataset, task, train_table, val_table, test_table)
#         dick[task] = tuple(task.entity_table, loader_dict, col_stats_dict, entity)

# for task, required_items in dick.items():
#     train(task, required_items) 
#     log loss/metrics
