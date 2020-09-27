#!/usr/bin/env python3
"""
    Scripts for evaluating the GPT-2 DST model prediction with Task2 metrics : bleu, retrieval 

    First, we parse the line-by-line stringified format into
    the structured DST output.

    We then run the main DST Evaluation script to get results.
"""
import argparse
import json
import nltk
import numpy as np

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

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

def eval_retrieval_score(candidate_path, predicted_path):

    f = open(candidate_path, 'r')
    cand_file = json.load(f)
    gen_file = open(predicted_path, 'r')

    gen_file_lines = gen_file.readlines() 
    input_sent_idx = 0
    gt_ranks = []
    for dialog in cand_file["retrieval_candidates"]:
        for round_ut in dialog["retrieval_candidates"]:
            #round_ut["turn_idx"]
            cand_list = round_ut["retrieval_candidates"] #list of 100 cand
            input_sent = gen_file_lines[input_sent_idx].strip()
            round_datum = []
            for cand in cand_list:
                round_datum.append(calculate_simmilarity(input_sent, cand_file["system_transcript_pool"][cand]))
            gt_score = round_datum[0]
            gt_ranks.append(np.sum(np.array(round_datum) > gt_score) + 1)
            input_sent_idx += 1

    gt_ranks = np.array(gt_ranks)

    result = {
        "r1": np.mean(gt_ranks <= 1),
        "r5": np.mean(gt_ranks <= 5),
        "r10": np.mean(gt_ranks <= 10),
        "mean": np.mean(gt_ranks),
        "mrr": np.mean(1/gt_ranks)

    }

    return result

def eval_bleu(target, predicted):
    def normalize_sen(sentence):
        return nltk.tokenize.word_tokenize(sentence.lower())

    with open(target, 'r') as f:
        # with open('furniture_devtest_dials_predicted_response.txt', 'r') as f2:
        with open(predicted, 'r') as f2:
            gt = json.load(f)
            ii = 0
            bleu_scores = []
            chencherry = nltk.translate.bleu_score.SmoothingFunction()
            for i in range(len(gt['dialogue_data'])):
                for j in range(len(gt['dialogue_data'][i]['dialogue'])):
                    gt_response = gt['dialogue_data'][i]['dialogue'][j]['system_transcript']
                    response = f2.readline().strip()
                    bleu_score = nltk.translate.bleu_score.sentence_bleu(
                        [normalize_sen(gt_response)],
                        normalize_sen(response),
                        smoothing_function=chencherry.method1
                    )
                    bleu_scores.append(bleu_score)
                    ii += 1

        return np.mean(bleu_scores)

if __name__ == '__main__':
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_target',
                        help='path for target, raw data (dialog.json file)' )
    parser.add_argument('--retrieval_candidate_path',
                        help='path for retrieval candidates')
    parser.add_argument('--input_path_predicted',
                        help='path for model prediction output, line-separated format (.txt)')
    parser.add_argument('--output_path_report',
                        help='path for saving evaluation summary (.json)')

    args = parser.parse_args()

    input_path_target = args.input_path_target
    retrieval_candidate_path = args.retrieval_candidate_path
    input_path_predicted = args.input_path_predicted
    output_path_report = args.output_path_report

    # import ipdb; ipdb.set_trace(context=10)
    bleu_score = eval_bleu(input_path_target, input_path_predicted)
    retrieval_score = eval_retrieval_score(retrieval_candidate_path, input_path_predicted)

    retrieval_score["bleu"] = bleu_score
    report =  retrieval_score

    # B : print results rightaway
    print(report)

    # Save report
    with open(output_path_report, 'w') as f_out:
        json.dump(report, f_out)



