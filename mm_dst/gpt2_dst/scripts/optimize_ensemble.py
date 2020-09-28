#!/usr/bin/env python3
"""
    Scripts for analyzing the GPT-2 DST model predictions.
"""
import argparse
import json
import ipdb 
import os
import itertools
from collections import Counter 
from gpt2_dst.utils.convert import parse_flattened_results_from_file
from utils.evaluate_dst import evaluate_from_flat_list

START_BELIEF_STATE = '=> Belief State :'

check = 0
def ensemble_turn(domain,prompt,predictions,turn_idx) :
    global output 
    global check 
    prompt = prompt.rstrip() 
    originalPrompt = prompt
    total_model = len(predictions)
    longest_frame_len = 0
    for model_idx in range(total_model) :  # for the current turn_idx, find the longest frame length
        if predictions[model_idx][turn_idx] : 
            longest_frame_len = max(longest_frame_len, len(predictions[model_idx][turn_idx]))

    frame_index = [0] * longest_frame_len
    for model_idx in range(total_model) :  
        if predictions[model_idx][turn_idx] : 
            frame_index[len(predictions[model_idx][turn_idx])-1] += 1 

    voted_frame_index = frame_index.index(max(frame_index)) + 1 
    
    for frame_idx in range(0,voted_frame_index) : 
        frame_list = []
        for model_idx in range(total_model) : 
            if predictions[model_idx][turn_idx] and len(predictions[model_idx][turn_idx]) > frame_idx : 
                frame_list.append(predictions[model_idx][turn_idx][frame_idx])
         
        ensembled = ensemble_frame(domain,frame_list)
        act = ensembled[0]
        slot = ensembled[1] 
        prompt+=(act + " " + "[" + slot + "] ")
    #if len(prompt) > 10 :
        #prompt = originalPrompt + "DA:NONE:NONE:NONE" + " [ ] "
    output.write(prompt + "<EOB>\n")

def ensemble_frame(domain,frame_list):    # compare frames generated from different models
    act_dict = [] 
    slot_dict = [] 
    longest_slot_size = 0
    slot_index = []
    slots = ""
    for frame in frame_list : 
        act_dict.append(frame['act'])
        longest_slot_size = max(longest_slot_size, len(frame['slots']))

    for i in range(longest_slot_size) : 
        slot_index.append(0)
    slot_index.append(0)
    for frame in frame_list : 
        slot_index[len(frame['slots'])] += 1 

    voted_slot_size = slot_index.index(max(slot_index)) 

    for frame in frame_list : 
        slot_dict = [] 
        for slot_index in range(voted_slot_size) : 
            slot = ""
            if frame['slots'] and len(frame['slots']) >= slot_index+1 : 
                if len(frame['slots'][slot_index]) >= 2 :  
                    slot = domain+"-" + frame['slots'][slot_index][0] + "=" +frame['slots'][slot_index][1]
            slot_dict.append(slot)
    d = Counter(slot_dict)     
    c = Counter(act_dict)
    i = 0 
    for slot in d.most_common(voted_slot_size) : 
        if i == 0 :
            slots = slots + " " + slot[0] 
        else : 
            slots = slots + " , " + slot[0] 
        i+=1
    return c.most_common(1)[0][0], slots

if __name__ == '__main__':
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_predicted_list', nargs='+',
                        help='list of path for predicted results from different models, seperated with a space') 
    parser.add_argument('--output_path_ensembled',
                        help='output file path for ensembled results (.txt')
    parser.add_argument('--prompts_from_file',
                        help='prompt file path (.txt')
    parser.add_argument('--domain',
                        help='domain (furniture or fashion) ', default="furniture")
    parser.add_argument('--target', help='target file for optimization')
    args = parser.parse_args()
    input_path_predicted_list = args.input_path_predicted_list  
    prompts_from_file = args.prompts_from_file
    output_path_ensembled = args.output_path_ensembled
    domain = args.domain 
    target = args.target 
    predictions = [] 
    if args.prompts_from_file:
        with open(args.prompts_from_file) as handle:
            prompts = handle.readlines()
    
    optimized_subset = [] 
    best_act_f1 = 0
    best_slot_f1 = 0  

    for L in range(2, len(input_path_predicted_list )+1):
        for subset in itertools.combinations(input_path_predicted_list , L):
            predictions = [] 
            for model in subset :
                if not model:
                    import ipdb; ipdb.set_trace(context=10)

                predicted = parse_flattened_results_from_file(model)
                predictions.append(predicted)

            turn_idx = 0 
            
            output  = open(output_path_ensembled,'w') 
            output.write("")
            output  = open(output_path_ensembled,'a') 

            for line in prompts :
                ensemble_turn(domain,line,predictions,turn_idx)
                turn_idx+=1  
            list_predicted = parse_flattened_results_from_file(output_path_ensembled)
            list_target = parse_flattened_results_from_file(target)
            # Evaluate
            report = evaluate_from_flat_list(list_target, list_predicted)
            slot_f1 = report['slot_f1']
            act_f1 = report['act_f1']
            if best_slot_f1 < slot_f1 : 
                best_slot_f1 = slot_f1 
                best_act_f1 = act_f1
                optimized_subset = subset[:]
    
    print("Best result => act f1 : {}, slot f1 : {}".format(best_act_f1, best_slot_f1))
    print("obtained with the combination of {} models..".format(optimized_subset))
    predictions = [] 
    output  = open(output_path_ensembled,'w') 
    output.write("")
    output  = open(output_path_ensembled,'a') 
    
    for model in optimized_subset : 
        predicted = parse_flattened_results_from_file(model)
        predictions.append(predicted)
    print("Optimized result is saved in {}..".format(output_path_ensembled))
