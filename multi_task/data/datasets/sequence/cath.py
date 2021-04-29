# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com

CATH 20201021 database
"""

import os
import pandas as pd
from torch.utils.data import Dataset

from typing import List, Tuple
import string
import itertools
from Bio import SeqIO

import numpy as np
import math

# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

def read_contact_map(ann_path):
    information_list = []
    with open(ann_path, "r") as f:
        for line in f:
            information = []
            raw_information = line.strip("\n").split(" ")
            for sub_info in raw_information:
                if sub_info != "":
                    information.append(sub_info)
            information_list.append(information)

    cm_shape = (int(information[0])+1, int(information[1])+1)
    information_numpy = np.array(information_list)
    contact_map = information_numpy[:, 2].reshape(cm_shape).astype(np.float)

    # normalization
    contact_map = np.minimum(contact_map / 100, 1)

    return contact_map


class CATH(Dataset):
    def __init__(self, root, dataset_type="sequence", set_name="train", transform=None):  #sequence  msa
        self.batch_converter = None
        self.data_type = dataset_type
        self.set_name = set_name
        self.sequence_dir = os.path.join(root, "dump_seq")
        self.annotation_dir = os.path.join(root, "dump_2d")
        #self.name_record_path = os.path.join(root, "dump_valid_list")
        save_path = "/home/liyu/cjy/esm/multi_task/data/datasets/cath"
        self.name_record_path = os.path.join(save_path, "split_dataset_" + set_name +".txt")
        self.transform = transform
        self.sequences, self.annotations, stats = self.__dataset_info(self.name_record_path, self.sequence_dir, self.annotation_dir)
        """
        self.statistics = pd.DataFrame(stats, columns=["Length", "CM_mean", "CM_max", "CM_min"])
        print("{} Dataset Info:".format(self.set_name))
        print("Length-Frequency Table")
        print(self.statistics["Length"].describe())#value_counts())
        print("CM_mean-Numerical Summaries")
        print(self.statistics["CM_mean"].describe())
        print("CM_max-Numerical Summaries")
        print(self.statistics["CM_max"].describe())
        print("CM_min-Numerical Summaries")
        print(self.statistics["CM_min"].describe())
        #"""

        self.only_obtain_label = False  # for class_balance_random_sampler tranverse dataset rapidly

    def __getitem__(self, index):

        if self.data_type == "sequence":
            data = read_sequence(self.sequences[index])
        elif self.data_type == "msa":
            data = read_msa(self.sequences[index], 64)

        contact_map = read_contact_map(self.annotations[index])

        annotation = (contact_map)

        return data, annotation, self.batch_converter

    def __len__(self):
        return len(self.sequences)

    def __dataset_info(self, name_record_path, sequence_dir, annotation_dir):
        sequences = []
        annotations = []
        statistics = []

        with open(name_record_path, "r") as f:
            for line in f:
                name = line.strip()
                sequence_path = os.path.join(sequence_dir, name + ".seq")

                length = len(read_sequence(sequence_path)[1])
                if length > 500:
                    continue

                sequences.append(sequence_path)
                annotation_path = os.path.join(annotation_dir, name + ".2d")
                annotations.append(annotation_path)

                """
                seq = read_sequence(sequence_path)
                length = len(seq[1])
                contact_map = read_contact_map(annotation_path)
                cm_max = contact_map.max()
                cm_mean = contact_map.mean()
                cm_min = contact_map.min()
                #if length > 500:
                #    continue
                statistics.append([length, cm_mean, cm_max, cm_min])
                print(len(statistics))
                """

                if os.path.exists(sequence_path) != True or os.path.exists(annotation_path) != True:
                    print("{} doesn't exist.".format(name))

        return sequences, annotations, statistics
