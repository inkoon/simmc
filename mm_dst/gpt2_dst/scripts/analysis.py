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
    with open(os.path.join(output_dir, "analysis.json"), 'w') as f_out:
        json.dump(acts, f_out)

def analyze_turn(turn_idx,true_turn, pred_turn):
    global wrong_act_wrong_slot
    global right_act_wrong_slot
    global wrong_act_right_slot
    global right_act_right_slot
    global no_prediction
    # Must preserve order in which frames appear.
    for frame_idx in range(len(true_turn)):
        # For each frame
        true_frame = true_turn[frame_idx]
        if frame_idx >= len(pred_turn): ## less prediction made 
            pred_frame = {}
            wrong_act_wrong_slot.add(turn_idx) ## wrong overall
            no_prediction+=1
        else:
            pred_frame = pred_turn[frame_idx]
            analyze_frame(turn_idx,true_frame,pred_frame)





def analyze_frame(idx,true_frame, pred_frame):
    """
        If strict=True,
            For each dialog_act (frame), set(slot values) must match.
            If dialog_act is incorrect, its set(slot values) is considered wrong.
    """
    global total_act
    global total_correct
    global acts 
    global k 
    global slots
    global wrong_act_wrong_slot
    global right_act_wrong_slot
    global right_act_right_slot
    global wrong_act_right_slot
    # Compare Dialog Actss
    flag = 0 
    true_act = true_frame['act'] if 'act' in true_frame else None
    pred_act = pred_frame['act'] if 'act' in pred_frame else None

    true_act_split = true_act.split(":")
    true_act_simple = true_act_split[1] 
    true_act_detail = ':'.join(true_act_split[2:])

    if pred_act is None : 
        wrong_act_wrong_slot.add(idx)
        return None 
    pred_act_split = pred_act.split(":")
    if pred_act != "ERR:CHITCHAT" and pred_act != "ERR:MISSING_CONTEXT" and len(pred_act_split) < 3 : 
        wrong_act_wrong_slot.add(idx)
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
        right_act_right_slot.add(idx)
        total_correct = total_correct + 1
        total_act = total_act + 1 
        acts[true_act_simple]['correct'] += 1
        if acts[true_act_simple]['detail'].get(true_act_detail) is None : 
            acts[true_act_simple]['detail'][true_act_detail]['correct'] = 1
        else : 
            acts[true_act_simple]['detail'][true_act_detail]['correct'] = acts[true_act_simple]['detail'][true_act_detail]['correct'] + 1
    else : 

        total_act = total_act + 1 
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
   
    if flag == 1 : ## if right DA, check slots 
        if true_frame_slot_values == pred_frame_slot_values : ## perfect prediction made 
            right_act_right_slot.add(idx)
        else : 
            right_act_wrong_slot.add(idx)
    else :
        if true_frame_slot_values == pred_frame_slot_values : 
            wrong_act_right_slot.add(idx)
        else :
            wrong_act_wrong_slot.add(idx)
         



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
    
    no_prediction = 0
    total_act = 0
    total_correct = 0
    acts = {

    }
   
    slots = {

    }

    k = 0

    wrong_act_wrong_slot = set()
    right_act_wrong_slot = set()
    right_act_right_slot = set() 
    wrong_act_right_slot = set() 

    analyze_from_flat_list(list_target, list_predicted)
    # Save report

    right_act_right_slot = right_act_right_slot - right_act_right_slot.intersection(wrong_act_wrong_slot) - right_act_right_slot.intersection(wrong_act_right_slot)
    wrong_act_wrong_slot = wrong_act_wrong_slot - wrong_act_wrong_slot.intersection(right_act_wrong_slot) - wrong_act_wrong_slot.intersection(wrong_act_right_slot) - wrong_act_wrong_slot.intersection(right_act_right_slot)
    right_act_wrong_slot = right_act_wrong_slot - right_act_wrong_slot.intersection(right_act_right_slot)
    wrong_act_right_slot = wrong_act_right_slot - wrong_act_right_slot.intersection(wrong_act_wrong_slot)
    results = []
    idx = -1
    
    all_wrong_f = open(os.path.join(output_dir, "all_wrong.txt"),'w') ## Wrong DA or slot
    all_wrong_f .write('')
    all_wrong_f  = open(os.path.join(output_dir, "all_wrong.txt"),'a') 
    right_act_wrong_slot_f = open(os.path.join(output_dir, "right_act_wrong_slot.txt"),'w')  ## Right DA but wrong slot 
    right_act_wrong_slot_f.write('')
    right_act_wrong_slot_f = open(os.path.join(output_dir, "right_act_wrong_slot.txt"),'a+') 
    all_correct_f = open(os.path.join(output_dir, "perfect.txt"),'w') 
    all_correct_f.write('')
    all_correct_f = open(os.path.join(output_dir, "perfect.txt"),'a+') 
    wrong_act_right_slot_f =  open(os.path.join(output_dir, "wrong_act_right_slot.txt"),'w')
    wrong_act_right_slot_f.write('') 
    wrong_act_right_slot_f =  open(os.path.join(output_dir, "wrong_act_right_slot.txt"),'a+') 

    with open(os.path.join(output_dir, "analysis.json"), 'w') as f_out:
        json.dump(acts, f_out)
    
    with open(os.path.join(output_dir,"dialogue_analysis.txt"),'w') as f_out :
        f_out.write("*************************************** DIALOG ACT ANALYSIS ***********************************\n")
        f_out.write("[ Resullts : {:.2f} % .... {} correct prediction made out of {} frames ] \n ".format(100*(total_correct/total_act),total_correct,total_act))
        f_out.write("no frame prediction made for {} times\n".format(no_prediction))
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

    turn = 0
    with open(input_path_target) as targetfile, open(input_path_predicted) as predictedfile : 
        for x, y in zip(targetfile, predictedfile):
            x = x.strip()
            y = y.strip()
            turn += 1
            target = x.split(START_BELIEF_STATE)
            predicted = y.split(START_BELIEF_STATE)
            idx+=1
            if idx in wrong_act_wrong_slot : 
                all_wrong_f.write("=============================DIALOGUE #{}====================\n".format(idx))
                all_wrong_f.write("{0}\n".format(target[0]))
                all_wrong_f.write("------------------------------------------------------------\n")
                all_wrong_f.write("Target : {0}\nPredicted : {1}\n\n".format(target[1],predicted[1]))
            if idx in right_act_wrong_slot : 
                right_act_wrong_slot_f.write("=============================DIALOGUE #{}====================\n".format(idx))
                right_act_wrong_slot_f.write("{0}\n".format(target[0]))
                right_act_wrong_slot_f.write("------------------------------------------------------------\n")
                right_act_wrong_slot_f.write("Target : {0}\nPredicted : {1}\n\n".format(target[1],predicted[1]))
            if idx in right_act_right_slot : 
                all_correct_f.write("=============================DIALOGUE #{}====================\n".format(idx))
                all_correct_f.write("{0}\n".format(target[0]))
                all_correct_f.write("------------------------------------------------------------\n")
                all_correct_f.write("Target : {0}\nPredicted : {1}\n\n".format(target[1],predicted[1]))
            if idx in wrong_act_right_slot : 
                wrong_act_right_slot_f.write("=============================DIALOGUE #{}====================\n".format(idx))
                wrong_act_right_slot_f.write("{0}\n".format(target[0]))
                wrong_act_right_slot_f.write("------------------------------------------------------------\n")
                wrong_act_right_slot_f.write("Target : {0}\nPredicted : {1}\n\n".format(target[1],predicted[1]))

    print("Out of {} total dialogues..".format(turn))
    print("{} right DA and slot predictions made... {:.2f}%".format(len(right_act_right_slot), 100*len(right_act_right_slot)/turn))
    print("{} right DA but wrong slot predictions....{:.2f}%".format(len(right_act_wrong_slot), 100*len(right_act_wrong_slot)/turn))
    print("{} wrong DA and right slot predictions....{:.2f}%".format(len(wrong_act_right_slot), 100*len(wrong_act_right_slot)/turn))
    print("{} wrong DA and wrong slot predictions....{:.2f}%\n".format(len(wrong_act_wrong_slot), 100*len(wrong_act_wrong_slot)/turn))
    print("Analysis completed..!\nDetailed analysis saved in {}".format(output_dir))
