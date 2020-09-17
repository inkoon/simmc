#!/usr/bin/env python3
"""
    Scripts for analyzing the GPT-2 DST model predictions.

"""
import argparse
import json
import ipdb 
import os
from gpt2_dst.utils.convert import parse_flattened_results_from_file
from utils.evaluate_dst import evaluate_from_flat_list

START_BELIEF_STATE = '=> Belief State :'

def analyze_from_flat_list(d_true, d_pred):

    # Count # corrects & # wrongs
    for turn_idx in range(len(d_true)):
        true_turn = d_true[turn_idx]
        pred_turn = d_pred[turn_idx]
        #ipdb.set_trace(context=20) # BREAKPOINT
        analyze_turn(turn_idx,true_turn, pred_turn)
    
    # Save report
    with open(os.path.join(output_dir, "analysis.txt"), 'w') as f_out:
        json.dump(acts, f_out)

def analyze_turn(turn_idx,true_turn, pred_turn):
    global wrong_index
    # Must preserve order in which frames appear.
    for frame_idx in range(len(true_turn)):
        # For each frame
        true_frame = true_turn[frame_idx]
        if frame_idx >= len(pred_turn):
            pred_frame = {}
            wrong_index.add(turn_idx)
        else:
            pred_frame = pred_turn[frame_idx]
            if analyze_frame(turn_idx,true_frame,pred_frame, strict=False) == 0 :
                wrong_index.add(turn_idx)


               




def analyze_frame(idx,true_frame, pred_frame, strict=True):
    """
        If strict=True,
            For each dialog_act (frame), set(slot values) must match.
            If dialog_act is incorrect, its set(slot values) is considered wrong.
    """
    global total_act
    global total_correct
    global acts 
    global slots
    global wrong_index
    global slot_part_right
    global perfect
    # Compare Dialog Actss
    flag = 0 
    total_act = total_act + 1
    true_act = true_frame['act'] if 'act' in true_frame else None
    pred_act = pred_frame['act'] if 'act' in pred_frame else None

    true_act_split = true_act.split(":")
    true_act_simple = true_act_split[1] 
    true_act_detail = ':'.join(true_act_split[2:])

    if pred_act is None : 
        return None 
    pred_act_split = pred_act.split(":")
    if pred_act != "ERR:CHITCHAT" and pred_act != "ERR:MISSING_CONTEXT" and len(pred_act_split) < 3 : 
        return None 
    pred_act_simple = pred_act_split[1] 
    pred_act_detail = ':'.join(pred_act_split[2:])
    #ipdb.set_trace(context=20) 
    if true_act_simple in acts : 
        acts[true_act_simple]['total'] =  acts[true_act_simple]['total'] + 1
        if true_act_detail in acts[true_act_simple]['detail'] :  ## already exists
            acts[true_act_simple]['detail'][true_act_detail]['total'] += 1
        else : 
            if not acts[true_act_simple]['detail'].get(true_act_detail) : 
                acts[true_act_simple]['detail'][true_act_detail] = {
                  'total' : 1,
                  'correct' : 0,
                  'all-wrong' : {

             },
             'part-wrong' : {

                } 
             }
    else :
        acts[true_act_simple] = {
            'total' : 1,
            'correct' : 0,
            'detail' : {

            }
        }
        acts[true_act_simple]['detail'][true_act_detail] = {
            'total' : 1,
            'correct' : 0,
            'all-wrong' : {

            },
            'part-wrong' : {

            } 
        }
    
    if true_act == pred_act : ## Act correctly predicted 
        flag = 1
        total_correct = total_correct + 1
        acts[true_act_simple]['correct'] += 1
        if acts[true_act_simple]['detail'].get(true_act_detail) is None : 
            acts[true_act_simple]['detail'][true_act_detail]['correct'] = 1
        else : 
            acts[true_act_simple]['detail'][true_act_detail]['correct'] = acts[true_act_simple]['detail'][true_act_detail]['correct'] + 1
    else : 
        #ipdb.set_trace(context=20) 
        if true_act_simple == pred_act_simple : ## Correct DA but wrong activities
            if not acts[true_act_simple]['detail'].get(true_act_detail)['part-wrong'] : 
                acts[true_act_simple]['detail'].get(true_act_detail)['part-wrong'] = { 
                        pred_act_detail : 1 
                }
            else :
                if pred_act_detail in acts[true_act_simple]['detail'][true_act_detail]['part-wrong'] : 
                    acts[true_act_simple]['detail'][true_act_detail]['part-wrong'][pred_act_detail] += 1
                else : 
                    acts[true_act_simple]['detail'][true_act_detail]['part-wrong'][pred_act_detail] = 1

        else : ## All wrong 
            if not acts[true_act_simple]['detail'].get(true_act_detail)['all-wrong'] : 
                acts[true_act_simple]['detail'].get(true_act_detail)['all-wrong'] = {
                    pred_act : 1 
                }
            else :
                if pred_act in acts[true_act_simple]['detail'][true_act_detail]['all-wrong'] : 
                    acts[true_act_simple]['detail'][true_act_detail]['all-wrong'][pred_act] += 1
                else :
                    acts[true_act_simple]['detail'][true_act_detail]['all-wrong'][pred_act] = 1
    


    # Compare Slots
    true_frame_slot_values = {f'{k}={v}' for k, v in true_frame.get('slots', [])}
    pred_frame_slot_values = {f'{k}={v}' for k, v in pred_frame.get('slots', [])}
    if len(true_frame_slot_values.intersection(pred_frame_slot_values)) > 0 :
        slot_part_right.add(idx)   ## partially right 

    if flag == 1 :
        if true_frame_slot_values == pred_frame_slot_values :
            perfect.add(idx)
        else :
            flag = 0 ## correct act but wrong slot values 
    
    return flag 



if __name__ == '__main__':
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_target',
                        help='path for target, line-separated format (.txt)')
    parser.add_argument('--input_path_predicted',
                        help='path for model prediction output, line-separated format (.txt)')
    parser.add_argument('--output_dir',
                        help='output directory for saving analysis summary files')
    parser.add_argument('--limit',
                        help='percentage', type=float,default=0.3)
                                    

    args = parser.parse_args()
    input_path_target = args.input_path_target
    input_path_predicted = args.input_path_predicted
    output_dir = args.output_dir
    limit = args.limit
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    list_target = parse_flattened_results_from_file(input_path_target)
    list_predicted = parse_flattened_results_from_file(input_path_predicted)
    total_act = 0
    total_correct = 0
    acts = {

    }
   
    slots = {

    }

    wrong_index = set()
    slot_part_right = set()
    perfect = set() 

    analyze_from_flat_list(list_target, list_predicted)
    
    # Save report

    results = []
    idx = -1
    with open(os.path.join(output_dir, "comparison.txt"),'w') as out:
        out.write('')
    with open(input_path_target) as targetfile, open(input_path_predicted) as predictedfile, open(os.path.join(output_dir, "comparison.txt"),"a+") as out: 
        for x, y in zip(targetfile, predictedfile):
            x = x.strip()
            y = y.strip()
            target = x.split(START_BELIEF_STATE)
            predicted = y.split(START_BELIEF_STATE)
            idx+=1
            if idx in wrong_index : 
                out.write("=============================DIALOGUE #{}====================\n".format(idx))
                out.write("{0}\n".format(target[0]))
                out.write("------------------------------------------------------------\n")
                out.write("Target : {0}\nPredicted : {1}\n\n".format(target[1],predicted[1]))


    with open(os.path.join(output_dir, "analysis.json"), 'w') as f_out:
        json.dump(acts, f_out)
    
    with open(os.path.join(output_dir,"dialogue_analysis.txt"),'w') as f_out :
        f_out.write("*************************************** DIALOG ACT ANALYSIS ***********************************\n")
        f_out.write("[ Resullts : {:.2f} % .... {} correct prediction made out of {} Dialog acts ] \n ".format(100*(total_correct/total_act),total_correct,total_act))
        for act in acts :
            f_out.write("=======================================DA:{}===========================================\n".format(act))
            f_out.write(" Accuracy : {:.2f} % =>  {} Correct prediction out of Total {}\n".format(100* acts[act]['correct']/acts[act]['total'], acts[act]['correct'],acts[act]['total']))
            for detail in acts[act]['detail']:
                f_out.write("\t{}\n".format(detail))
                total = acts[act]['detail'][detail]['total']
                correct = acts[act]['detail'][detail]['correct']
                wrong = total - correct
                f_out.write("\t\t=>  {} Correct prediction out of Total {}\n".format(correct,total))
                for part_wrong in acts[act]['detail'][detail]['part-wrong'] :
                    if acts[act]['detail'][detail]['part-wrong'][part_wrong] >= wrong * limit :
                        num_part_wrong = acts[act]['detail'][detail]['part-wrong'][part_wrong]
                        f_out.write("\t\t\t Wrongly predicted {} for {} times ( {:.2f}% )\n".format(part_wrong,num_part_wrong,100*(num_part_wrong/wrong)))
                for all_wrong in acts[act]['detail'][detail]['all-wrong'] : 
                    if acts[act]['detail'][detail]['all-wrong'][all_wrong] >= wrong * limit :
                        num_part_wrong = acts[act]['detail'][detail]['all-wrong'][all_wrong]
                        f_out.write("\t\t\t Wrongly predicted {} for {} times ({:.2f}% )\n".format(all_wrong, num_part_wrong, 100*(num_part_wrong/wrong) )) 
                f_out.write("--------------------------------------------------------------------------------------\n")
            f_out.write("\n\n\n")


