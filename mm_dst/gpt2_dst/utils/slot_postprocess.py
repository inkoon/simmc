import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

predicted = open(args.path + 'furniture_devtest_dials_predicted.txt', 'r')

predicted_processed = open(args.path + "furniture_devtest_dials_predicted_processed.txt", 'w')

act_path = open("../../gpt2_dst/data/act.json", 'r')
slot_path = open("../../gpt2_dst/data/slot.json", 'r')

act_list = json.load(act_path)
slot_list = json.load(slot_path)

BELIEF_STATE = " => Belief State : "
EOB = " <EOB> "

token_match_path = open("./special_token_match.json", 'r')
token_match = json.load(token_match_path)

l = []

def postprocess(reader, writer):
    for i, line in enumerate(reader.readlines()):
        # writer.write(str(i) + '\t')
        split = line.split(BELIEF_STATE)
        prompt = split[0]
        bs = split[1]
        
        split = bs.split(EOB)
        state = split[0]
        response = split[1]

        writer.write(prompt)
        writer.write(BELIEF_STATE)
        
        state = state.replace("  ", " ")
        state = state.replace("   ", " ")
        for key in token_match.keys():
            if key in state:
                state = state.replace(key, token_match[key])

        writer.write(state)

        writer.write(EOB)
        writer.write(response)
        l.append(i)

# postprocess(target, target_processed)
postprocess(predicted, predicted_processed)


