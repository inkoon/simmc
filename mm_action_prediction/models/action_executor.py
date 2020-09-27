"""Executes the actions and predicts action attributes for SIMMC.

Author(s): Satwik Kottur
"""


from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import json
import torch
import torch.nn as nn

import loaders
import models
from tools import torch_support as support


class ActionExecutor(nn.Module):
    def __init__(self, params):
        """Initialize classifiers.
        """
        super(ActionExecutor, self).__init__()
        self.params = params
        self.belief_state_size = int(self.params["hidden_size"]/32)
        self.slot_size = 2
        input_size = self.params["hidden_size"]
        if self.params["text_encoder"] == "transformer":
            input_size = self.params["word_embed_size"]
        self.action_net = self._get_classify_network(input_size, params["num_actions"])
        if params["use_action_attention"]:
            self.attention_net = models.SelfAttention(input_size)
        # If multimodal input state is to be used.
        if self.params["use_multimodal_state"]:
            input_size += self.params["hidden_size"]
        self.action_net = self._get_classify_network(input_size, params["num_actions"])
        self.params["use_belief_state"]=False
        # B : If belief state is to be used.
        if self.params["use_belief_state"]:
            # one belief state
            input_size += self.belief_state_size
            # two belief states
            input_size += self.belief_state_size
            # slot
            input_size += 6*self.slot_size
            #import ipdb;ipdb.set_trace(context=10)
            if self.params["text_encoder"] == "transformer" and self.belief_state_size%2==1:
                input_size -= 2
        self.action_net = self._get_classify_network(input_size, params["num_actions"])
        # Read action metadata.
        with open(params["metainfo_path"], "r") as file_id:
            action_metainfo = json.load(file_id)["actions"]
            action_dict = {ii["name"]: ii["id"] for ii in action_metainfo}
            self.action_metainfo = {ii["name"]: ii for ii in action_metainfo}
            self.action_map = loaders.Vocabulary(immutable=True, verbose=False)
            sorted_actions = sorted(action_dict.keys(), key=lambda x: action_dict[x])
            self.action_map.set_vocabulary_state(sorted_actions)
        # Read action attribute metadata.
        with open(params["attr_vocab_path"], "r") as file_id:
            self.attribute_vocab = json.load(file_id)
        # Create classifiers for action attributes.
        self.classifiers = {}
        for key, val in self.attribute_vocab.items():
            self.classifiers[key] = self._get_classify_network(
                input_size, len(val)
            )
        self.classifiers = nn.ModuleDict(self.classifiers)

        # All attributes list
        if params["domain"] == "furniture":
            self.all_classifier_list = [
                'furnitureType',
                'color',
                'matches',
                'navigate_direction',
                'position',
                'direction'
            ]
        elif params["domain"] == "fashion":
            self.all_classifier_list = [
                'attributes'
            ]

        # Model multimodal state.
        if params["use_multimodal_state"]:
            if params["domain"] == "furniture":
                self.multimodal_embed = models.CarouselEmbedder(params)
            elif params["domain"] == "fashion":
                self.multimodal_embed = models.UserMemoryEmbedder(params)
            else:
                raise ValueError("Domain neither of furniture/fashion")

        # NOTE: Action output is modeled as multimodal state.
        if params["use_action_output"]:
            if params["domain"] == "furniture":
                self.action_output_embed = models.CarouselEmbedder(params)
            elif params["domain"] == "fashion":
                self.action_output_embed = models.UserMemoryEmbedder(params)
            else:
                raise ValueError("Domain neither of furniture/fashion")
        self.criterion_mean = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.criterion_multi = torch.nn.MultiLabelSoftMarginLoss()


        # B : belief state embedding
        self.action_embedding = nn.Embedding(params["action_num"], int(self.belief_state_size/2), padding_idx=0)
        self.attribute_embedding = nn.Embedding(params["attribute_num"], int(self.belief_state_size/2), padding_idx=0)
        self.slot_embedding = nn.Embedding(params["slot_num"], int(self.slot_size), padding_idx=0)



    def forward(self, batch, prev_outputs):
        """Forward pass a given batch.

        Args:
            batch: Batch to forward pass
            prev_outputs: Output from previous modules.

        Returns:
            outputs: Dict of expected outputs
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        outputs = {}
        if self.params["use_action_attention"] and self.params["encoder"] != "tf_idf":
            encoder_state = prev_outputs["hidden_states_all"]
            batch_size, num_rounds, max_len = batch["user_mask"].shape
            encoder_mask = batch["user_utt"].eq(batch["pad_token"])
            encoder_mask = support.flatten(encoder_mask, batch_size, num_rounds)
            encoder_state = self.attention_net(encoder_state, encoder_mask)
        else:
            encoder_state = prev_outputs["hidden_state"][0][-1]

        encoder_state_old = encoder_state
        # Multimodal state.
        if self.params["use_multimodal_state"]:
            if self.params["domain"] == "furniture":
                encoder_state = self.multimodal_embed(
                    batch["carousel_state"],
                    encoder_state,
                    batch["dialog_mask"].shape[:2]
                )
            elif self.params["domain"] == "fashion":
                multimodal_state = {}
                for ii in ["memory_images", "focus_images"]:
                    multimodal_state[ii] = batch[ii]
                encoder_state = self.multimodal_embed(
                    multimodal_state, encoder_state, batch["dialog_mask"].shape[:2]
                )

        # B : belief state.
        if self.params["use_belief_state"]:
            #import ipdb;ipdb.set_trace(context=10)
            batch_act_0 = batch["belief_state_act"][:,:num_rounds,0].to(device)
            batch_attr_0 = batch["belief_state_attr"][:,:num_rounds,0].to(device)
            batch_act_1 = batch["belief_state_act"][:,:num_rounds,1].to(device)
            batch_attr_1 = batch["belief_state_attr"][:,:num_rounds,1].to(device)
            slot_0_0 = batch["belief_state_slot"][:,:num_rounds,0,0].to(device)
            slot_0_1 = batch["belief_state_slot"][:,:num_rounds,0,1].to(device)
            slot_0_2 = batch["belief_state_slot"][:,:num_rounds,0,2].to(device)
            slot_1_0 = batch["belief_state_slot"][:,:num_rounds,1,0].to(device)
            slot_1_1 = batch["belief_state_slot"][:,:num_rounds,1,1].to(device)
            slot_1_2 = batch["belief_state_slot"][:,:num_rounds,1,2].to(device)

            #import ipdb;ipdb.set_trace(context=7)
            batch_act_0 = support.flatten(batch_act_0, batch_size, num_rounds)
            batch_attr_0 = support.flatten(batch_attr_0, batch_size, num_rounds)
            batch_act_1 = support.flatten(batch_act_1, batch_size, num_rounds)
            batch_attr_1 = support.flatten(batch_attr_1, batch_size, num_rounds)
            batch_slot_0_0 = support.flatten(slot_0_0, batch_size, num_rounds)
            batch_slot_0_1 = support.flatten(slot_0_1, batch_size, num_rounds)
            batch_slot_0_2 = support.flatten(slot_0_2, batch_size, num_rounds)
            batch_slot_1_0 = support.flatten(slot_1_0, batch_size, num_rounds)
            batch_slot_1_1 = support.flatten(slot_1_1, batch_size, num_rounds)
            batch_slot_1_2 = support.flatten(slot_1_2, batch_size, num_rounds)

            act_0_embedd = self.action_embedding(batch_act_0).to(device)
            attr_0_embedd = self.attribute_embedding(batch_attr_0).to(device)
            act_1_embedd = self.action_embedding(batch_act_1).to(device)
            attr_1_embedd = self.attribute_embedding(batch_attr_1).to(device)
            slot_0_0_embedd = self.slot_embedding(batch_slot_0_0).to(device)
            slot_0_1_embedd = self.slot_embedding(batch_slot_0_1).to(device)
            slot_0_2_embedd = self.slot_embedding(batch_slot_0_2).to(device)
            slot_1_0_embedd = self.slot_embedding(batch_slot_1_0).to(device)
            slot_1_1_embedd = self.slot_embedding(batch_slot_1_1).to(device)
            slot_1_2_embedd = self.slot_embedding(batch_slot_1_2).to(device)


            belief_state_0 = torch.cat((act_0_embedd, attr_0_embedd, slot_0_0_embedd, slot_0_1_embedd, slot_0_2_embedd), dim=1)
            belief_state_1 = torch.cat((act_1_embedd, attr_1_embedd, slot_1_0_embedd, slot_1_1_embedd, slot_1_2_embedd), dim=1)
            #import ipdb;ipdb.set_trace(context=10)
            encoder_state = torch.cat((encoder_state, belief_state_0, belief_state_1), dim=1)

        # Predict and execute actions.
        #if self.params["use_belief_state"]:
            #action_net_1 = self._get_classify_network(2*self.params["hidden_size"]+2*belief_state_0.shape[1], self.params["num_actions"]).to(device)
            #action_logits = action_net_1(encoder_state.to(device)).to(device)
        #else:
        #import ipdb;ipdb.set_trace(context=10)
        action_logits = self.action_net(encoder_state);#print("encoder_state+_shape: " +str(encoder_state.shape))
        dialog_mask = batch["dialog_mask"];#print("action_logits__shape: " +str(action_logits.shape))
        batch_size, num_rounds = dialog_mask.shape
        loss_action = self.criterion(action_logits, batch["action"].view(-1)); #print("action"+str(batch["action"].shape))
        loss_action.masked_fill_((~dialog_mask).view(-1), 0.0)
        loss_action_sum = loss_action.sum() / dialog_mask.sum().item()
        outputs["action_loss"] = loss_action_sum
        if not self.training:
            # Check for action accuracy.
            action_logits = support.unflatten(action_logits, batch_size, num_rounds)
            actions = action_logits.argmax(dim=-1)
            action_logits = nn.functional.log_softmax(action_logits, dim=-1)
            action_list = self.action_map.get_vocabulary_state()
            # Convert predictions to dictionary.
            action_preds_dict = [
                {
                    "dialog_id": batch["dialog_id"][ii].item(),
                    "predictions": [
                        {
                            "action": self.action_map.word(actions[ii, jj].item()),
                            "action_log_prob": {
                                action_token: action_logits[ii, jj, kk].item()
                                for kk, action_token in enumerate(action_list)
                            },
                            "attributes": {},
                            "turn_id": jj
                        }
                        for jj in range(batch["dialog_len"][ii])
                    ]
                }
                for ii in range(batch_size)
            ]
            outputs["action_preds"] = action_preds_dict
        else:
            actions = batch["action"]

        # Run classifiers based on the action, record supervision if training.
        if self.training:
            assert (
                "action_super" in batch
            ), "Need supervision to learn action attributes"
        attr_logits = collections.defaultdict(list)
        attr_loss = collections.defaultdict(list)
        encoder_state_unflat = support.unflatten(
            encoder_state, batch_size, num_rounds
        )

        host = torch.cuda if self.params["use_gpu"] else torch
        for inst_id in range(batch_size):
            for round_id in range(num_rounds):
                # Turn out of dialog length.
                if not dialog_mask[inst_id, round_id]:
                    continue

                cur_action_ind = actions[inst_id, round_id].item()
                cur_action = self.action_map.word(cur_action_ind)
                cur_state = encoder_state_unflat[inst_id, round_id]
                supervision = batch["action_super"][inst_id][round_id]

                # Predict attributes on all actions for ensemble.
                if not self.training:
                    action_pred_datum = action_preds_dict[inst_id]["predictions"][round_id]
                    action_pred_datum["attributes_prob"] = {}
                    for key in self.all_classifier_list:
                        classifier = self.classifiers[key]
                        attr_dict = {}
                        model_preds = classifier(cur_state)
                        for pred_i, pred in enumerate(model_preds):
                            attr_dict[self.attribute_vocab[key][pred_i]] = float(pred)
                        action_pred_datum["attributes_prob"][key] = attr_dict

                # If there is no supervision, ignore and move on to next round.
                if supervision is None:
                    continue

                # Run classifiers on attributes.
                # Attributes overlaps completely with GT when training.
                if self.training:
                    classifier_list = self.action_metainfo[cur_action]["attributes"]
                    if self.params["domain"] == "furniture":
                        for key in classifier_list:
                            cur_gt = (
                                supervision.get(key, None)
                                if supervision is not None
                                else None
                            )
                            new_entry = (cur_state, cur_gt, inst_id, round_id)
                            attr_logits[key].append(new_entry)
                    elif self.params["domain"] == "fashion":
                        for key in classifier_list:
                            cur_gt = supervision.get(key, None)
                            gt_indices = host.FloatTensor(
                                len(self.attribute_vocab[key])
                            ).fill_(0.)
                            gt_indices[cur_gt] = 1
                            new_entry = (cur_state, gt_indices, inst_id, round_id)
                            attr_logits[key].append(new_entry)
                    else:
                        raise ValueError("Domain neither of furniture/fashion!")
                else:
                    classifier_list = self.action_metainfo[cur_action]["attributes"]
                    if self.params["domain"] == "furniture":
                        # Predict attributes based on the predicted action.
                        for key in classifier_list:
                            classifier = self.classifiers[key]
                            model_pred = classifier(cur_state).argmax(dim=-1)
                            attr_pred = self.attribute_vocab[key][model_pred.item()]
                            action_pred_datum["attributes"][key] = attr_pred
                    elif self.params["domain"] == "fashion":
                        # Predict attributes based on predicted action.
                        for key in classifier_list:
                            classifier = self.classifiers[key]
                            model_pred = classifier(cur_state) > 0
                            attr_pred = [
                                self.attribute_vocab[key][index]
                                for index, ii in enumerate(model_pred)
                                if ii
                            ]
                            action_pred_datum["attributes"][key] = attr_pred
                    else:
                        raise ValueError("Domain neither of furniture/fashion!")

        # Compute losses if training, else predict.
        if self.training:
            for key, values in attr_logits.items():
                #import ipdb;ipdb.set_trace(context=10)
                classifier = self.classifiers[key]
                prelogits = [ii[0] for ii in values if ii[1] is not None]
                if not prelogits:
                    continue
                logits = classifier(torch.stack(prelogits, dim=0))
                if self.params["domain"] == "furniture":
                    gt_labels = [ii[1] for ii in values if ii[1] is not None]
                    gt_labels = host.LongTensor(gt_labels)
                    attr_loss[key] = self.criterion_mean(logits, gt_labels)
                elif self.params["domain"] == "fashion":
                    gt_labels = torch.stack(
                        [ii[1] for ii in values if ii[1] is not None], dim=0
                    )
                    attr_loss[key] = self.criterion_multi(logits, gt_labels)
                else:
                    raise ValueError("Domain neither of furniture/fashion!")

            total_attr_loss = host.FloatTensor([0.0])
            if len(attr_loss.values()):
                total_attr_loss = sum(attr_loss.values()) / len(attr_loss.values())
            outputs["action_attr_loss"] = total_attr_loss

        # Obtain action outputs as memory cells to attend over.
        if self.params["use_action_output"]:
            if self.params["domain"] == "furniture":
                encoder_state_out = self.action_output_embed(
                    batch["action_output"],
                    encoder_state_old,
                    batch["dialog_mask"].shape[:2],
                )
            elif self.params["domain"] == "fashion":
                multimodal_state = {}
                for ii in ["memory_images", "focus_images"]:
                    multimodal_state[ii] = batch[ii]
                # For action output, advance focus_images by one time step.
                # Output at step t is input at step t+1.
                feature_size = batch["focus_images"].shape[-1]
                zero_tensor = host.FloatTensor(batch_size, 1, feature_size).fill_(0.)
                multimodal_state["focus_images"] = torch.cat(
                    [batch["focus_images"][:, 1:, :], zero_tensor], dim=1
                )
                encoder_state_out = self.multimodal_embed(
                    multimodal_state, encoder_state_old, batch["dialog_mask"].shape[:2]
                )
            else:
                raise ValueError("Domain neither furniture/fashion!")
            outputs["action_output_all"] = encoder_state_out

        outputs.update(
            {"action_logits": action_logits, "action_attr_loss_dict": attr_loss}
        )
        return outputs

    def _get_classify_network(self, input_size, num_classes):
        """Construct network for predicting actiosn and attributes.
        """
        return nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, num_classes),
        )
