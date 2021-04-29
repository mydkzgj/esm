# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com

same as text data?
"""

import os
import pandas as pd
from torch.utils.data import Dataset

from typing import List, Tuple
import string
import itertools
from Bio import SeqIO

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


class General_Sequence(Dataset):
    def __init__(self, root, dataset_type="sequence", set_name="train", transform=None):
        self.batch_converter = None
        self.data_type = dataset_type
        self.set_name = set_name
        self.sequence_dir = os.path.join(root)
        #self.annotation_path = os.path.join(root, "Split_Dataset", "Once", set_name + ".txt")
        self.transform = transform
        self.sequence_names = ["1a3a_1_A.a3m", "5ahw_1_A.a3m", "1xcr_1_A.a3m"]
        self.sequences = [os.path.join(self.sequence_dir, s_name) for s_name in self.sequence_names]
        """
        self.sequences, self.labels, stats = self.__dataset_info(self.annotation_path, self.sequence_dir)
        self.statistics = pd.DataFrame(stats, columns=["Label", "Length"])
        print("{} Dataset Info:".format(self.set_name))
        print("Label-Frequency Table")
        print(self.statistics["Label"].value_counts())
        print("Length-Numerical Summaries")
        print(self.statistics["Length"].describe())
        #"""

        self.only_obtain_label = False  # for class_balance_random_sampler tranverse dataset rapidly

    def __getitem__(self, index):

        if self.data_type == "sequence":
            data = read_sequence(self.sequences[index])
        elif self.data_type == "msa":
            data = read_msa("1a3a_1_A.a3m", 64)

        return data, self.batch_converter

    def __len__(self):
        return len(self.sequences)

    def __dataset_info(self, record_txt, sequence_dir):
        sequences = []
        labels = []
        statistic = []

        with open(record_txt, "r") as f:
            for line in f:
                s, l = line.strip().split(" ")
                sequence_filename = os.path.join(sequence_dir, s)
                sequences.append(sequence_filename)
                labels.append(self.label_map[l])

                length = int(s.split(".csv")[0].split("_")[-1])
                statistic.append([self.label_map[l], length])

        return sequences, labels, statistic







