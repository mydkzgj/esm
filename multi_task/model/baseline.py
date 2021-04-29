# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

import torch

from .backbones import choose_backbone
from .auxiliary_modules import choose_auxiliary_module
from .weights_init import *
from ptflops import get_model_complexity_info

class Baseline(nn.Module):
    def __init__(self,
                 backbone_name,
                 contact_predictor_name="none",
                 secondary_structure_predictor_name="none",
                 ):
        super(Baseline, self).__init__()
        # 0.Configuration
        self.backbone_name = backbone_name
        self.contact_predictor_name = contact_predictor_name
        self.secondary_structure_predictor_name = secondary_structure_predictor_name

        # 1.Build backbone
        self.backbone, self.backbone_batch_converter = choose_backbone(self.backbone_name)

        # 2.Build Contact Predictor
        args = {}
        args["in_features"] = self.backbone.args.layers * self.backbone.args.attention_heads
        args["prepend_bos"] = self.backbone.prepend_bos
        args["append_eos"] = self.backbone.append_eos
        args["eos_idx"] = self.backbone.eos_idx
        self.ct_predictor = choose_auxiliary_module(self.contact_predictor_name, args)

        # 3.Build Secondary Structure Predictor
        args = {}
        self.ss_predictor = choose_auxiliary_module(self.secondary_structure_predictor_name, args)

        # 5.Initialization of parameters
        self.backbone.apply(weights_init_kaiming)
        self.backbone.apply(weights_init_classifier) # maybe with classifier itself
        if self.ct_predictor is not None:
            self.ct_predictor.apply(weights_init_classifier)
        if self.ss_predictor is not None:
            self.ct_predictor.apply(weights_init_classifier)

    def forward(self, x, need_head_weights=True):
        with torch.no_grad():
            results = self.backbone(x, need_head_weights=need_head_weights)

        if self.ct_predictor is not None:
            attentions = results["attentions"]
            results["contacts"] = self.ct_predictor(x, attentions)

        if self.ss_predictor is not None:
            #results["secondary_structure_predictions"] = self.ct_predictor(embeddings)
            print(1)

        return results

    def contact_to_rg_vectors(self, pd_contacts, gt_contacts):
        rg_logits = []
        rg_labels = []
        for index, gt_contact in enumerate(gt_contacts):
            pd_contact = pd_contacts[index][0:gt_contact.shape[0], 0:gt_contact.shape[1]]
            rg_logits.append(pd_contact.flatten())
            rg_labels.append(gt_contact.flatten())
        rg_logits = torch.cat(rg_logits, dim=0)
        rg_labels = torch.cat(rg_labels, dim=0)

        return rg_logits, rg_labels

    def contact_to_cf_vectors(self, pd_contacts, gt_contacts):
        rg_logits = []
        rg_labels = []
        for index, gt_contact in enumerate(gt_contacts):
            pd_contact = pd_contacts[index][0:gt_contact.shape[0], 0:gt_contact.shape[1]]
            rg_logits.append(pd_contact.flatten())
            rg_labels.append(gt_contact.gt(0.07).float().flatten())
        rg_logits = torch.cat(rg_logits, dim=0)
        rg_labels = torch.cat(rg_labels, dim=0)

        return rg_logits, rg_labels

    # load parameter
    def load_param(self, load_choice, model_path):
        param_dict = torch.load(model_path)["model"]
        if load_choice == "Base":
            base_dict = self.backbone.state_dict()
            for i in param_dict:
                module_name = i.replace("sentence_encoder.", "").replace("encoder.", "")
                if module_name not in self.backbone.state_dict():
                    print("Cannot load %s, Maybe you are using incorrect framework"%i)
                    continue
                self.backbone.state_dict()[module_name].copy_(param_dict[i])

        elif load_choice == "Overall":
            overall_dict = self.state_dict()
            for i in param_dict:
                if i in self.state_dict():
                    self.state_dict()[i].copy_(param_dict[i])
                elif "base."+i in self.state_dict():
                    self.state_dict()["base."+i].copy_(param_dict[i])
                elif "backbone."+i in self.state_dict():
                    self.state_dict()["backbone."+i].copy_(param_dict[i])
                elif i.replace("base", "backbone") in self.state_dict():
                    self.state_dict()[i.replace("base", "backbone")].copy_(param_dict[i])
                else:
                    print("Cannot load %s, Maybe you are using incorrect framework"%i)
                    continue

        print("Complete Load Weight")

    def count_param(model, input_shape=(3, 224, 224)):
        with torch.cuda.device(0):
            flops, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            return ('{:<30}  {:<8}'.format('Computational complexity: ', flops)) + (
                '{:<30}  {:<8}'.format('Number of parameters: ', params))