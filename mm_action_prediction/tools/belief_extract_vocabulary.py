"""Extracts vocabulary for SIMMC dataset.

Author: Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import argparse
from nltk.tokenize import word_tokenize
from transformers import GPT2Tokenizer


def main(args):
    # Read the data, parse the datapoints.
    print("Reading: {}".format(args["train_json_path"]))
    with open(args["train_json_path"], "r") as file_id:
        train_data = json.load(file_id)
    dialog_data = train_data["dialogue_data"]

    # Load pretrained GPT2Tokenizer
    if args["gpt2"]:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    counts = {}
    intent_list = []
    intent_has_none = 0
    attr_list = ['none','unk']
    slot_list = ['none']
    stack_id=0
    if "task3_fusion_path" in args:
        with open(args["task3_fusion_path"], "r") as file_id:
            belief_train = json.load(file_id)
        if len(belief_train) > len(train_data["dialogue_data"]):
            for datum_id, datum in enumerate(train_data["dialogue_data"]):
                datum_list = []
                for i in range(len(datum["dialogue"])):
                    datum_list.append(belief_train[stack_id])
                    stack_id+=1
                my_data_list.append(datum_list)
            belief_train = my_data_list
 
        for utter_id, uttr in enumerate(belief_train): # data 1개
            for bs_id, belief_states in enumerate(uttr):
                for bs in belief_states:
                    if bs[0]==None:
                        continue
                    else:
                        if bs[0][0]=="E":
                            if bs[0].split('.')[0] not in intent_list:
                                intent_list.append(bs[0].split('.')[0])
                        elif bs[0][0]=="D":
                            bel_list = bs[0].split('.')
                            if bel_list[0] not in intent_list:
                                intent_list.append(bel_list[0])
                            if len(bel_list)>=2 and bel_list[1] not in attr_list:
                                attr_list.append(bel_list[1])
                        else:
                            raise KeyboardInterrupt('undefined belief_state')
                    if bs[1]==None or len(bs[1])==0:
                        continue
                    else:
                        for slot in bs[1]:
                            if '_' in slot:
                                slot=slot.split('_')[0]
                            if slot not in slot_list:
                                slot_list.append(slot)

    for datum in dialog_data:
        dialog_utterances = [
            ii[key] for ii in datum["dialogue"]
            for key in ("transcript", "system_transcript")
        ]
        if args["gpt2"]:
            dialog_tokens = [
                tokenizer.tokenize(ii.lower()) for ii in dialog_utterances
            ]
        else:
            dialog_tokens = [
                word_tokenize(ii.lower()) for ii in dialog_utterances
            ]
        for turn in dialog_tokens:
            for word in turn:
                counts[word] = counts.get(word, 0) + 1
        if "task3_fusion_path" in args:
            continue

        for turn in datum["dialogue"]:
            # act information
            for num in range(len(turn["belief_state"])):
                if turn["belief_state"][num]["act"]==None and intent_has_none==0:
                    intent_list.append('none')
                    intent_has_none = 1
                else:
                    if turn["belief_state"][num]["act"][0]=="E":
                        if turn["belief_state"][num]["act"] not in intent_list:
                            intent_list.append(turn["belief_state"][num]["act"] )
                    elif turn["belief_state"][num]["act"][0]=="D":
                        bel_list = turn["belief_state"][num]["act"].split('.')
                        if bel_list[0] not in intent_list:
                            intent_list.append(bel_list[0])
                        if len(bel_list)>=2 and bel_list[1] not in attr_list:
                            attr_list.append(bel_list[1])
                    else:
                        raise KeyboardInterrupt('undefined belief_state')
                # slot information
                if len(turn["belief_state"][num]["slots"])==0:
                    pass
                else:
                    for slot in turn["belief_state"][num]["slots"]:
                        my_slot = slot[0]#.split('-')[-1]
                        if '_' in my_slot:
                            my_slot = my_slot.split('_')[0]
                        if my_slot not in slot_list:
                            slot_list.append(my_slot)
 
    intent_list.sort()
    intent_dict = {}
    slot_dict={}
    for i, intent in enumerate(intent_list):
        intent_dict[i+1]=intent
    attr_list.sort()
    attr_dict={}
    for i, attr in enumerate(attr_list):
        attr_dict[i+1]=attr
    for i, slot in enumerate(slot_list):
        slot_dict[i+1]=slot
    # Add <pad>, <unk>, <start>, <end>.
    counts["<pad>"] = args["threshold_count"] + 1
    counts["<unk>"] = args["threshold_count"] + 1
    counts["<start>"] = args["threshold_count"] + 1
    counts["<end>"] = args["threshold_count"] + 1

    word_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    words = [ii[0] for ii in word_counts if ii[1] >= args["threshold_count"]]
    # Save answers and vocabularies.
    vocabulary = {"word": words,  "act_type" : intent_dict, "attr_type" : attr_dict, "slot_type" : slot_dict}
    print("Identified {} words..".format(len(words)))
    print("Saving dictionary: {}".format(args["vocab_save_path"]))
    if args['gpt2']:
        tokenizer.save_vocabulary(args["vocab_save_path"])
        with open(f'{args["vocab_save_path"]}gpt2_vocab.json', "w") as file_id:
            json.dump(vocabulary, file_id)
    else:
        with open(f'{args["vocab_save_path"]}_vocabulary.json', "w") as file_id:
            json.dump(vocabulary, file_id)


if __name__ == "__main__":
    # Read the commandline arguments.
    parser = argparse.ArgumentParser(description="Extract vocabulary")
    parser.add_argument(
        "--train_json_path",
        default="data/furniture_data.json",
        help="Path to read the vocabulary (train) JSON",
    )
    parser.add_argument(
        "--vocab_save_path",
        default="data/furniture_vocabulary.json",
        help="Path to read the vocabulary (train) JSON",
    )
    parser.add_argument(
        "--threshold_count",
        default=0,
        type=int,
        help="Words are included if beyond this threshold",
    )
    parser.add_argument(
        "--gpt2",
        action="store_true",
        default=False,
        help="Use gpt2 tokenizer"
    }
    parser.add_argument(
        "--task3_fusion_path",
        deault=None
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
