#!/usr/bin/env python3
"""
    Scripts for analyzing the GPT-2 DST model predictions.

"""
import argparse
import json
import ipdb 
from gpt2_dst.utils.convert import parse_flattened_results_from_file
from utils.evaluate_dst import evaluate_from_flat_list

acts = {

}
   
slot = {

}


def analyze_from_flat_list(d_true, d_pred):
  

    # Count # corrects & # wrongs
    for turn_idx in range(len(d_true)):
        true_turn = d_true[turn_idx]
        pred_turn = d_pred[turn_idx]
        #ipdb.set_trace(context=20) # BREAKPOINT
        analyze_turn(true_turn, pred_turn)
    
    # Save report
    with open(output_path_report, 'w') as f_out:
        json.dump(acts, f_out)

def analyze_turn(true_turn, pred_turn):

    # Must preserve order in which frames appear.
    for frame_idx in range(len(true_turn)):
        # For each frame
        true_frame = true_turn[frame_idx]
        if frame_idx >= len(pred_turn):
            pred_frame = {}
        else:
            pred_frame = pred_turn[frame_idx]
        analyze_frame(true_frame, pred_frame, strict=False)



def analyze_frame(true_frame, pred_frame, strict=True):
    """
        If strict=True,
            For each dialog_act (frame), set(slot values) must match.
            If dialog_act is incorrect, its set(slot values) is considered wrong.
    """

    # Compare Dialog Actss
    true_act = true_frame['act'] if 'act' in true_frame else None
    pred_act = pred_frame['act'] if 'act' in pred_frame else None

    true_act_split = true_act.split(":")
    true_act_simple = true_act_split[1] 
    true_act_detail = ':'.join(true_act_split[2:])

    if pred_act is None : 
        return None 
    pred_act_split = pred_act.split(":")
    if pred_act != "ERR:CHITCHAT" and len(pred_act_split) < 3 : 
        print(pred_act)
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
    



if __name__ == '__main__':
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_target',
                        help='path for target, line-separated format (.txt)')
    parser.add_argument('--input_path_predicted',
                        help='path for model prediction output, line-separated format (.txt)')
    parser.add_argument('--output_path_report',
                        help='path for saving evaluation summary (.json)')
    parser.add_argument('--limit',
                        help='percentage', type=float,default=0.3)
                         

    args = parser.parse_args()
    input_path_target = args.input_path_target
    input_path_predicted = args.input_path_predicted
    output_path_report = args.output_path_report
    limit = args.limit
    # Convert the data from the GPT-2 friendly format to JSON
    list_target = parse_flattened_results_from_file(input_path_target)
    list_predicted = parse_flattened_results_from_file(input_path_predicted)

    # Evaluate
    analyze_from_flat_list(list_target, list_predicted)
    
    # Save report
    #with open(output_path_report, 'w') as f_out:
     #   json.dump(acts, f_out)
    print("*************************************** GENERATED ANALAYSIS ***********************************\n")
    i = 0
    for act in acts :
        i+=1
        print("=======================================DA:{}===========================================".format(act))
        print(" Accuracy : {:.2f} % =>  {} Correct prediction out of Total {}".format(100* acts[act]['correct']/acts[act]['total'], acts[act]['correct'],acts[act]['total']))
        for detail in acts[act]['detail']:
            print("\t{}".format(detail))
            total = acts[act]['detail'][detail]['total']
            correct = acts[act]['detail'][detail]['correct']
            wrong = total - correct
            print("\t\t=>  {} Correct prediction out of Total {}".format(correct,total))
            for part_wrong in acts[act]['detail'][detail]['part-wrong'] :
                if acts[act]['detail'][detail]['part-wrong'][part_wrong] >= wrong * limit :
                    print("\t\t\t Wrongly predicted {} for {} times".format(part_wrong,acts[act]['detail'][detail]['part-wrong'][part_wrong]))
            for all_wrong in acts[act]['detail'][detail]['all-wrong'] : 
                if acts[act]['detail'][detail]['all-wrong'][all_wrong] >= wrong * limit :
                    #ipdb.set_trace(context=20) 
                    print("\t\t\t Wrongly predicted {} for {} times".format(all_wrong, acts[act]['detail'][detail]['all-wrong'][all_wrong])) 
            print("--------------------------------------------------------------------------------------")
        print("\n")

        

