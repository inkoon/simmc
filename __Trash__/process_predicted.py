import json
import argparse

f = open('./furniture_devtest_dials_predicted.txt', 'r')
w = open('./belief_state.json','w')

act_path = open('./act.json', 'r')
act_list = json.load(act_path)

slot_path = open('./slot.json', 'r')
slot_list = json.load(slot_path)

BELIEF_STATE = " => Belief State : "
EOB = " <EOB> "

belief_states = list()

def postprocess(reader):
    for line in reader.readlines():
        states = list()

        split = line.split(BELIEF_STATE)
        prompt = split[0]
        bs = split[1]
        
        split = bs.split(EOB)
        state = split[0]
        response = split[1]

        state_split = state.split(' ')
        state_act = list()
        state_attribute = list()
        state_slot = list()
        while '' in state_split:
            state_split.remove('')
        for i, token in enumerate(state_split):
            if token in act_list:
                if '.' in token:
                    sp = token.split('.')
                    state_act = sp[0]
                    state_attribute = sp[1]
                else:
                    state_act = token
                    state_attribute = 'none'
                states.append([state_act, state_attribute])
                
            # if token == '[':
                # if state_split[i+1]==']':
                    # state_slot = 
                # state_slot = state_split[i+1]
                # states.append([state_act, state_attribute, state_slot])

        belief_states.append(states)

postprocess(f)
belief_states_json = json.dumps(belief_states)
w.write(belief_states_json)
