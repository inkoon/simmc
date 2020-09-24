import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--domain', type=str, required=True)
args = parser.parse_args()

domain = args.domain

predicted = open(args.path + domain + '_devtest_dials_predicted.txt', 'r')

predicted_processed = open(args.path + domain + "_devtest_dials_predicted_processed.txt", 'w')

act_path = open(f"./{domain}/act.json", 'r')
slot_path = open(f"./{domain}/slot.json", 'r')

act_list = json.load(act_path)
slot_list = json.load(slot_path)

BELIEF_STATE = " => Belief State : "
EOB = " <EOB> "

token_match_path = open(f"./{domain}/token_to_special.json", 'r')
token_match = json.load(token_match_path)

l = []

def postprocess(reader, writer):
    for i, line in enumerate(reader.readlines()):
        # writer.write(str(i) + '\t')
        
        split = line.split(BELIEF_STATE)
        if len(split) != 2:
            print(f"ERROR : more than one belief state in line {i}!!")
            exit(1)
        else:
            prompt = split[0]
            bs = split[1]
            
        split = bs.split(EOB)
        
        if len(split) < 2:
            state = split[0].strip()
            response = '\n'
        elif len(split) > 2:
            state = split[0]
            response = split[1] + '\n'
        else:
            state = split[0]
            response = split[1]
            if response == "" or response == " " or response == "  ":
                response = '\n'

        state = state.replace("  ", " ")
        state = state.replace("   ", " ")
        for key in sorted(token_match, key=len, reverse=True):
            if key in state:
                state = state.replace(key, token_match[key])


        writer.write(prompt)
        writer.write(BELIEF_STATE)
        writer.write(state)
        writer.write(EOB)
        writer.write(response)
        l.append(i)

# postprocess(target, target_processed)
postprocess(predicted, predicted_processed)


