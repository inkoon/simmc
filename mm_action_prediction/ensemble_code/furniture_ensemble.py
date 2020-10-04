import json
import numpy as np
import argparse

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tools.action_evaluation as ae
import tools.response_evaluation as re
import tools.retrieval_evaluation as rre


###furniture ensemble

#model predictions file
def main(Model_types, best_gen_model_type):
    Tasks = ['task1', 'task2_r']
    #Model_types = ['HAE_R300', 'HRE_R300_MAG', 'HRE_R300_MMI', 'MN_R300_MAG', 'MN_R300_MMI'] 
    #best_gen_model_type = 'HRE_R300'

    action_model = []
    ret_model = []

    for task in Tasks:
        for model in Model_types:
            if task == "task1":
                action_model.append(json.load(open(f"./outputs/furniture/{model}/checkpoints/{task}_predict.json", "r")))
            elif task == "task2_r":
                ret_model.append(json.load(open(f"./outputs/furniture/{model}/checkpoints/{task}_predict.json", "r")))

    best_gen_model = json.load(open(f"./outputs/furniture/{best_gen_model_type}/checkpoints/task2_g_predict.json", "r"))

    """
    #action answer file
    gt_action = open("../data/simmc_furniture/furniture_devtest_dials_api_calls.json", "r")
    gt_action_file = json.load(gt_action)
    #generation answer file
    gt_responses = open("../data/simmc_furniture/furniture_devtest_dials.json", "r")
    gt_responses_file = json.load(gt_responses)
    #retrieval candidate file
    candidates = open("../data/simmc_furniture/furniture_devtest_dials_retrieval_candidates.json", "r")
    candidates_file = json.load(candidates)
    """

    def sum_action_logits(base_model, add_model, action_flag=False):
        action_att_dict = {"SearchFurniture":["color", "furnitureType"], "SpecifyInfo":["matches"], "FocusOnFurniture":["position"], "Rotate":["direction"], "NavigateCarousel":["navigate_direction"], "AddToCart":[], "None":[]}

        for a_i, action in enumerate(add_model["model_actions"]):
            for p_i, prediction in enumerate(action["predictions"]):
                add_pre = prediction["action_log_prob"]
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["SearchFurniture"] += add_pre["SearchFurniture"]
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["SpecifyInfo"] += add_pre["SpecifyInfo"]
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["FocusOnFurniture"] += add_pre["FocusOnFurniture"]
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["Rotate"] += add_pre["Rotate"]
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["NavigateCarousel"] += add_pre["NavigateCarousel"]
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["AddToCart"] += add_pre["AddToCart"]
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["None"] += add_pre["None"]
                
                if action_flag == True:
                    ac_dict = base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]
                    base_model["model_actions"][a_i]["predictions"][p_i]["action"]=str(max(ac_dict.keys(), key=(lambda k:ac_dict[k])))
                
                attribute_list = action_att_dict[base_model["model_actions"][a_i]["predictions"][p_i]["action"]]
                new_att_dict = {}
                for att in attribute_list:
                    if att == "furnitureType":
                        add_fur = prediction["attributes_prob"]["furnitureType"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["furnitureType"]["Accent Chairs"] += add_fur["Accent Chairs"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["furnitureType"]["Area Rugs"] += add_fur["Area Rugs"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["furnitureType"]["Bookcases"] += add_fur["Bookcases"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["furnitureType"]["Coffee & Cocktail Tables"] += add_fur["Coffee & Cocktail Tables"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["furnitureType"]["Dining Chairs"] += add_fur["Dining Chairs"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["furnitureType"]["Dining Tables"] += add_fur["Dining Tables"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["furnitureType"]["End Tables"] += add_fur["End Tables"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["furnitureType"]["Kitchen Islands"] += add_fur["Kitchen Islands"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["furnitureType"]["Office Chairs"] += add_fur["Office Chairs"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["furnitureType"]["Ottomans"] += add_fur["Ottomans"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["furnitureType"]["Sofas"] += add_fur["Sofas"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["furnitureType"]["Table Lamps"] += add_fur["Table Lamps"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["furnitureType"]["Teen Bookcases"] += add_fur["Teen Bookcases"]
                        if action_flag == True:
                            at_temp = base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["furnitureType"]
                            new_att_dict["furnitureType"] = str(max(at_temp.keys(), key=(lambda k: at_temp[k])))
                    elif att == "color":
                        add_col = prediction["attributes_prob"]["color"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["color"][""] += add_col[""]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["color"]["Beige"] += add_col["Beige"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["color"]["Black"] += add_col["Black"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["color"]["Blue"] += add_col["Blue"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["color"]["Brown"] += add_col["Brown"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["color"]["Gray"] += add_col["Gray"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["color"]["Green"] += add_col["Green"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["color"]["Purple"] += add_col["Purple"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["color"]["Red"] += add_col["Red"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["color"]["White"] += add_col["White"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["color"]["Yellow"] += add_col["Yellow"]
                        if action_flag == True:
                            at_temp = base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["color"]
                            new_att_dict["color"] = str(max(at_temp.keys(), key=(lambda k: at_temp[k])))
                    elif att == "matches":
                        add_mat = prediction["attributes_prob"]["matches"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["matches"]["color"] += add_mat["color"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["matches"]["dimensions"] += add_mat["dimensions"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["matches"]["info"] += add_mat["info"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["matches"]["material"] +=add_mat["material"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["matches"]["price"] += add_mat["price"]
                        if action_flag == True:
                            at_temp = base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["matches"]
                            new_att_dict["matches"] = str(max(at_temp.keys(), key=(lambda k: at_temp[k])))
                    elif att == "position":
                        add_pos = prediction["attributes_prob"]["position"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["position"]["center"] += add_pos["center"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["position"]["left"] += add_pos["left"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["position"]["right"] += add_pos["right"]
                        if action_flag == True:
                            at_temp = base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["position"]
                            new_att_dict["position"] = str(max(at_temp.keys(), key=(lambda k: at_temp[k])))
                    elif att == "direction":
                        add_dir = prediction["attributes_prob"]["direction"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["direction"]["back"] += add_dir["back"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["direction"]["down"] += add_dir["down"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["direction"]["front"] += add_dir["front"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["direction"]["left"] += add_dir["left"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["direction"]["right"] += add_dir["right"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["direction"]["up"] += add_dir["up"]
                        if action_flag == True:
                            at_temp = base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["direction"]
                            new_att_dict["direction"] = str(max(at_temp.keys(), key=(lambda k: at_temp[k])))
                    elif att == "navigate_direction":
                        add_nd = prediction["attributes_prob"]["navigate_direction"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["navigate_direction"]["Here"] += add_nd["Here"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["navigate_direction"]["Next"] += add_nd["Next"]
                        base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["navigate_direction"]["Previous"] += add_nd["Previous"]
                        if action_flag == True:
                            at_temp = base_model["model_actions"][a_i]["predictions"][p_i]["attributes_prob"]["navigate_direction"]
                            new_att_dict["navigate_direction"] = str(max(at_temp.keys(), key=(lambda k: at_temp[k])))
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
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["SearchFurniture"] /= total_num
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["SpecifyInfo"] /=total_num
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["FocusOnFurniture"] /=total_num
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["Rotate"] /=total_num
                base_model["model_actions"][a_i]["predictions"][p_i]["action_log_prob"]["NavigateCarousel"] /=total_num
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




    subtask1_out = open("./dstc9-simmc-teststd-furniture-subtask-1.json", "w")
    subtask2_gen_out = open("./dstc9-simmc-teststd-furniture-subtask-2-generation.json", "w")
    subtask2_ret_out = open("./dstc9-simmc-teststd-furniture-subtask-2-retrieval.json", "w")

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
