# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

import esm

def choose_backbone(backbone_name):
    # 1.ESM1b
    if backbone_name == 'esm1b':
        backbone, backbone_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        backbone_batch_converter = backbone_alphabet.get_batch_converter()

    # 2.MSA Transformer
    elif backbone_name == 'msa_transformer':
        backbone, backbone_alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
        backbone_batch_converter = backbone_alphabet.get_batch_converter()

    else:
        raise Exception("Wrong Backbone Type!")

    return backbone, backbone_batch_converter