import json
import argparse
import ipdb 
import numpy as np 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

BELIEF_STATE = " => Belief State : "
EOB = " <EOB> "

def calculate_simmilarity(input_sent, cand_sent):
    # tokenization 
    X_list = word_tokenize(input_sent)  
    Y_list = word_tokenize(cand_sent) 
  
    # sw contains the list of stopwords 
    sw = stopwords.words('english')  
    l1 =[];l2 =[] 
  
    # remove stop words from the string 
    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 
  
    # form a set containing keywords of both strings  
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
  
    # cosine formula  
    for i in range(len(rvector)): 
        c+= l1[i]*l2[i] 
    cosine = c / (float((sum(l1)*sum(l2))**0.5)+0.000001) 
    return cosine


def postprocess_generation(lines, dialogue_id, all_candidates,  system_transcript_pool) :
    output = {
        "dialog_id" : dialog_id, 
        "predictions" : [

        ]
    }
    scores = {
        "dialog_id" : dialog_id ,
        "candidate_scores": [

        ]
    }

    turn_id = 0 
    for line in lines : 
        prediction = {
        "response" : "",
        "turn_id" : turn_id
        }
        retrieval_score = {
            "scores": [], 
            "turn_id": turn_id
        }

        candidates = all_candidates[turn_id]["retrieval_candidates"]
        response =""
        parse = line.split(BELIEF_STATE)[1]
        generated = parse.split(EOB)
        if EOB in line : 
            response = generated[1].lstrip().strip()
        for candidate in candidates :
            system_utt = system_transcript_pool[candidate]
            retrieval_score["scores"].append(calculate_simmilarity(system_utt,response)*100)
        prediction["response"] = response
        output["predictions"].append(prediction)
        scores["candidate_scores"].append(retrieval_score)
        turn_id += 1 
    return output , scores

def find_candidates(dialog_id,all_candidates) :
    for candidate in all_candidates :
        if candidate["dialogue_idx"] == dialog_id :
            return candidate
    
    return None 

if __name__ == "__main__":
    # Parse input args")
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--domain', type=str, required=True)
    parser.add_argument('--dials_path',type=str,required=True)
    parser.add_argument('--retrieval_candidate_path', required=True)
    parser.add_argument('--data',type=str,required=True)
    args = parser.parse_args()

    predicted = open(args.path + args.domain+ '_' + args.data + '_dials_predicted.txt', 'r')
    predicted.seek(0)
    predicted_processed_generation = open(args.path + "dstc9-simmc-teststd-" + args.domain + "-subtask-2-generation.json", 'w')
    predicted_processed_retrieval = open(args.path + "dstc9-simmc-teststd-" + args.domain + "-subtask-2-retrieval.json", 'w')
    dials_path = args.dials_path
    domain = args.domain 
    retrieval_candidate= args.retrieval_candidate_path

    idx=0
    dialog_ids = [] 
    turn_length = []
    response_result = [] 
    retrieval_result = [] 
    candidates = [] 
    with open(args.retrieval_candidate_path,"r") as file :
        system_transcript_pool = json.load(file)["system_transcript_pool"]
    with open(args.retrieval_candidate_path,"r") as file :
        all_candidates = json.load(file)["retrieval_candidates"]
    
    with open(dials_path, "r") as file:
        dials = json.load(file)
        for dialogue in dials['dialogue_data'] : 
            leng = len(dialogue['dialogue'])
            dialog_ids.append(dialogue['dialogue_idx'])
            turn_length.append(leng)
    
    i = 0  
    for dialog_id in dialog_ids : 
        lines = []
        candidates = find_candidates(dialog_id,all_candidates)['retrieval_candidates']
        for t in range(turn_length[i]) :
            try:
                lines.append(next(predicted))
            except StopIteration:
                pass
        result = postprocess_generation(lines,dialog_id,candidates, system_transcript_pool)
        response_result.append(result[0])
        retrieval_result.append(result[1])
        i+=1
        #print("Finished converting dialog id : {}".format(dialog_id))
      
    
    json.dump(response_result, predicted_processed_generation)
    json.dump(retrieval_result, predicted_processed_retrieval)
    predicted_processed_generation.close()
    predicted_processed_retrieval.close()
    predicted.close()
    print("Done converting {} total dialog to task 2 output format for".format(i,domain))
