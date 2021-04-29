from .sequence import build_sequence_datasets

def build_datasets(dataset_name, dataset_type, cfg, for_train):
    if "sequence" in dataset_type:
        train_set, val_set, test_set, classes_list = build_sequence_datasets(dataset_name, dataset_type, cfg, for_train)
    else:
        raise Exception("Can not build {} type dataset {}".format(dataset_type, dataset_name))

    return train_set, val_set, test_set, classes_list

