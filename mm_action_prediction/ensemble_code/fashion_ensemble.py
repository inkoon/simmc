import json
import numpy as np
import argparse

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tools.action_evaluation as ae
import tools.response_evaluation as re
import tools.retrieval_evaluation as rre

###fashion ensemble

#model predictions file
def main(Model_types, best_gen_model_type):
    Tasks = ['task1', 'task2_g', 'task2_r']
    #Model_types = ['T_HAE_G300_TD', 'MN_R300_MAG_TD', 'MN_G300_TD'] 

    action_model = []
    ret_model = []

    for task in Tasks:
        for model in Model_types:
            if task == "task1":
                action_model.append(json.load(open(f"./outputs/fashion/{model}/checkpoints/{task}_predict.json", "r")))
            elif task == "task2_r":
                ret_model.append(json.load(open(f"./outputs/fashion/{model}/checkpoints/{task}_predict.json", "r")))
    
    best_gen_model = json.load(open(f"./outputs/fashion/{best_gen_model_type}/checkpoints/task2_g_predict.json", "r"))        
    """
    #action answer file
    gt_action = open("../data/simmc_fashion/fashion_devtest_dials_api_calls.json", "r")
    gt_action_file = json.load(gt_action)
    #generation answer file
    gt_responses = open("../data/simmc_fashion/fashion_devtest_dials.json", "r")
    gt_responses_file = json.load(gt_responses)
    #retrieval candidate file
    candidates = open("../data/simmc_fashion/fashion_devtest_dials_retrieval_candidates.json", "r")
    candidates_file = json.load(candidates)
    """

    def sum_action_logits(base_model, add_model, action_flag=False):
        action_att_dict = {"SearchMemory":["attributes"], "SearchDatabase":["attributes"], "SpecifyInfo":["attributes"], "AddToCart":[], "None":[]}

        for a_i, action in enumerate(add_model["model_actions"]):
            for p_i, prediction in enumerate(action["predictions"]):
                add_pre = prediction["action_log_prob"]
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["SearchMemory"] += add_pre["SearchMemory"]
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["SpecifyInfo"] += add_pre["SpecifyInfo"]
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["SearchDatabase"] += add_pre["SearchDatabase"]
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["AddToCart"] += add_pre["AddToCart"]
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["None"] += add_pre["None"]
            
                if action_flag == True:
                    ac_dict = base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]
                    base_model["model_actions"][a_i]["predictions"][p_i]["action"]=str(max(ac_dict.keys(), key=(lambda k:ac_dict[k])))
                
                attribute_list = action_att_dict[base_model["model_actions"][a_i]["predictions"][p_i]["action"]]
                new_att_dict = {"attributes":[]}
                for att in attribute_list:
                    if att == "attributes":
                        add_fur = prediction["attributes_prob"]["attributes"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["attributes"]["availableSizes"] += add_fur["availableSizes"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["attributes"]["brand"] += add_fur["brand"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["attributes"]["color"] += add_fur["color"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["attributes"]["customerRating"] += add_fur["customerRating"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["attributes"]["info"] += add_fur["info"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["attributes"]["other"] += add_fur["other"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["attributes"]["price"] += add_fur["price"]
                        if action_flag == True:
                            at_temp = base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["attributes"]
                            new_att_dict["attributes"].append(str(max(at_temp.keys(), key=(lambda k: at_temp[k]))))
                if action_flag == True:
                    base_model["model_actions"][a_i]["predictions"][p_i]["attributes"] = new_att_dict
                

        return base_model

    def sum_cand_scores(base_model, add_model):
        for cs_i, cand_score in enumerate(add_model["candidate_scores"]):
            for c_i, cand in enumerate(cand_score["candidate_scores"]):
                for i, c in enumerate(cand["scores"]):
                    base_model["candidate_scores"][cs_i]["candidate_scores"][c_i]["scores"][i] += c
        return base_model            

    def mean_action_logits(base_model, total_num):
        for a_i, action in enumerate(base_model["model_actions"]):
            for p_i, prediction in enumerate(action["predictions"]):
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["SearchMemory"] /= total_num
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["SpecifyInfo"] /=total_num
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["SearchDatabase"] /=total_num
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["AddToCart"] /=total_num
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["None"] /=total_num


    #model1에 전부 sum
    #action
    for i in range(1,len(Model_types)):
        if i == len(Model_types)-1:
            action_model[0] = sum_action_logits(action_model[0], action_model[i], action_flag=True)
        else:
            action_model[0] = sum_action_logits(action_model[0], action_model[i])

    mean_action_logits(action_model[0], len(Model_types))

    #retrieval
    for j in range(1, len(Model_types)):
        ret_model[0] = sum_cand_scores(ret_model[0], ret_model[i])

    ##

    #matches에 아래 형식에 맞게 넣어주기
    matches = {"model_response":[], "candidate_scores":[], "action_preds":[]}

    #attribute도 generation과 같아서 생각 좀 해봐야함
    matches["action_preds"] = action_model[0]["model_actions"]
    #generation은 bleu제일 높은 모델로, voting방법 생각 좀 해봐야함
    matches["model_response"] = best_gen_model["model_responses"]
    #ret
    matches["candidate_scores"] = ret_model[0]["candidate_scores"]

    
    # Compute BLEU score.
    model_responses = [jj for jj in matches["model_response"]]
    #bleu_score = re.evaluate_response_generation(gt_responses_file, model_responses)

    ###model_responses = None
    ###bleu_score = -1.

    # Evaluate retrieval score.
    candidate_scores = [jj for jj in matches["candidate_scores"]]
    #retrieval_metrics = rre.evaluate_response_retrieval(candidates_file, candidate_scores)
    #print(retrieval_metrics)

    ###retrieval_metrics = {}

    # Evaluate action prediction.
    action_predictions = [jj for jj in matches["action_preds"]]
    """
    action_metrics = ae.evaluate_action_prediction(gt_action_file, action_predictions)
    #print(action_metrics["confusion_matrix"])
    print_str = (
        #"\nEvaluation\n\tLoss: {:.2f}\n\t"
        #"Perplexity: {:.2f}\n\t"
        "BLEU: {:.3f}\n\t"
        "Action: {:.2f}\n\t"
        "Action Perplexity: {:.2f}\n\t"
        "Action Attribute Accuracy: {:.2f}"
    )
    print(
        print_str.format(
         #   avg_loss_eval,
         #   math.exp(avg_loss_eval),
            bleu_score,
            100 * action_metrics["action_accuracy"],
            action_metrics["action_perplexity"],
            100 * action_metrics["attribute_accuracy"]
        )
    )
    """

    subtask1_out = open("./dstc9-simmc-teststd-fashion-subtask-1.json", "w")
    subtask2_gen_out = open("./dstc9-simmc-teststd-fashion-subtask-2-generation.json", "w")
    subtask2_ret_out = open("./dstc9-simmc-teststd-fashion-subtask-2-retrieval.json", "w")

    json.dump(action_predictions, subtask1_out)
    json.dump(model_responses, subtask2_gen_out)
    json.dump(candidate_scores, subtask2_ret_out)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_types', '-m', nargs='+' ,help='Example) HAE_R300', dest='modelname')
    parser.add_argument('--best_gen', '-b', help='Example) HAE_R300', dest='bestgen')

    Model_types = parser.parse_args().modelname
    best_gen_model_type = parser.parse_args().bestgen

    return Model_types, best_gen_model_type

if __name__ == '__main__':
    Model_types, best_gen_model_type = get_arguments()
    main(Model_types, best_gen_model_type)

