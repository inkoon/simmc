import json
import nltk
import numpy as np

def normalize_sen(sentence):
    return nltk.tokenize.word_tokenize(sentence.lower())

with open('data/simmc_furniture/furniture_devtest_dials.json', 'r') as f:
    # with open('furniture_devtest_dials_predicted_response.txt', 'r') as f2:
    with open('furniture_devtest_dials_predicted_response_large.txt', 'r') as f2:
        gt = json.load(f)
        ii = 0
        bleu_scores = []
        chencherry = nltk.translate.bleu_score.SmoothingFunction()
        for i in range(len(gt['dialogue_data'])):
            for j in range(len(gt['dialogue_data'][i]['dialogue'])):
                gt_response = gt['dialogue_data'][i]['dialogue'][j]['system_transcript']
                response = f2.readline().split('\t')[1].strip()
                bleu_score = nltk.translate.bleu_score.sentence_bleu(
                    [normalize_sen(gt_response)],
                    normalize_sen(response),
                    smoothing_function=chencherry.method1
                )
                bleu_scores.append(bleu_score)
                ii += 1

    print(np.mean(bleu_scores))
