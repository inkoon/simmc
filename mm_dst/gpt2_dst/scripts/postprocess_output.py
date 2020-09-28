import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--domain', type=str, required=True)
parser.add_argument('--data', type=str, required=True)
args = parser.parse_args()

domain = args.domain
data = args.data

predicted = open(args.path + domain + '_' + data + '_dials_predicted.txt', 'r')

predicted_processed = open(args.path + "dstc9-simmc-" + data + 'std-' + domain + "-subtask-3.txt", 'w')


act_path = open(f"gpt2_dst/data/{domain}/act.json", 'r')
slot_path = open(f"gpt2_dst/data/{domain}/slot.json", 'r')

act_list = json.load(act_path)
slot_list = json.load(slot_path)

BELIEF_STATE = " => Belief State : "
EOB = " <EOB> "

token_match_path = open(f"gpt2_dst/data/{domain}/token_to_special.json", 'r')
token_match = json.load(token_match_path)

def postprocess(reader, writer):
    for i, line in enumerate(reader.readlines()):
        # writer.write(str(i) + '\t')
        
        split = line.split(BELIEF_STATE)
        if len(split) != 2:
            prompt = split[0]
            bs = split[-1]
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

postprocess(predicted, predicted_processed)
predicted_processed.close()
predicted.close()

domain = args.domain
data = args.data

source = open(args.path + "dstc9-simmc-" + data + 'std-' + domain + "-subtask-3.txt", 'r')
belief_state_path = open(args.path + domain + '_' + data + '_belief_state.json', 'w')

def make_input_for_task1(reader, writer):
    belief_state_list = []
    for i, line in enumerate(reader.readlines()):
        
        split = line.split(BELIEF_STATE)
        if len(split) != 2:
            prompt = split[0]
            bs = split[-1]
        else:
            prompt = split[0]
            bs = split[1]
            
        split = bs.split(EOB)
        states = split[0]
        states = states.replace("  ", " ")
        states = states.replace("   ", " ")

        belief_states = []
        for state in states.strip().split(']'):
            if state == '':
                continue
            bs = []
            slot = []
            for token in state.strip().split(' '):
                if token in act_list:
                    bs.append(token)
                if token in slot_list:
                    slot.append(token)
            bs.append(slot)
            belief_states.append(bs)
        belief_state_list.append(belief_states)

    bs_json = json.dumps(belief_state_list)
    writer.write(bs_json)

make_input_for_task1(source, belief_state_path)
source.close()
belief_state_path.close()
