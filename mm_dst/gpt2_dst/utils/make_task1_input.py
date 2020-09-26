import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--domain', type=str, required=True)
args = parser.parse_args()
domain = args.domain

source = open(args.path + domain + '_devtest_dials_predicted.txt', 'r')

target = open(args.path + domain + '_devtest_belief_state.json', 'w')


act_path = open(f"gpt2_dst/utils/{domain}/act.json", 'r')
slot_path = open(f"gpt2_dst/utils/{domain}/slot.json", 'r')

act_list = json.load(act_path)
slot_list = json.load(slot_path)

BELIEF_STATE = " => Belief State : "
EOB = " <EOB> "

token_match_path = open(f"gpt2_dst/utils/{domain}/token_to_special.json", 'r')
token_match = json.load(token_match_path)

l = []

def postprocess(reader, writer):
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

# postprocess(target, target_processed)
postprocess(source, target)


