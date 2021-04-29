import random
import os

def Once_Split_Dataset(dataset, proportion={"train":5, "valid":1, "test":1}):
    """
    :param dataset:
    :param proportion:
    :return:
    """
    sum = 0
    for p in proportion.keys():
        sum += proportion[p]

    split_dataset = {}
    start_pos = 0
    for index, set_key in enumerate(proportion.keys()):
        if index == len(proportion.keys()) - 1:
            end_pos = 1
        else:
            end_pos = start_pos + proportion[set_key] / sum
        split_dataset[set_key] = {}
        split_dataset[set_key]["pos"] = [start_pos, end_pos]
        start_pos = end_pos

    for label_key in dataset.keys():
        random.shuffle(dataset[label_key])

        l = len(dataset[label_key])
        for set_key in proportion.keys():
            start_pos = int(l * split_dataset[set_key]["pos"][0])
            end_pos = int(l * split_dataset[set_key]["pos"][1])
            split_dataset[set_key][label_key] = dataset[label_key][start_pos: end_pos]
    return split_dataset

def split_dataset(root_path, anntation_name, save_path):
    txt_path = os.path.join(root_path, anntation_name)
    f = open(txt_path, "r")
    dataset = {"all":[]}
    for line in f:
        dataset["all"].append(line.strip("\n"))
    f.close()

    proportion = {"train": 5, "valid": 1, "test": 1}
    split_datasets = Once_Split_Dataset(dataset, proportion)
    for key in proportion.keys():
        f = open(os.path.join(save_path, "split_dataset_" + key+".txt"), "w")
        for name in split_datasets[key]["all"]:
            f.write("{}\n".format(name))
        f.close()


if __name__ == "__main__":
    root_path = "/mnt/datadisk0/shengwwang/databases/"
    anntation_name = "dump_valid_list"
    save_path = "/home/liyu/cjy/esm/multi_task/data/datasets/cath"
    if os.path.exists(save_path) != True:
        os.makedirs(save_path)
    split_dataset(root_path, anntation_name, save_path)