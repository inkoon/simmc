import random

f = open("./../data/furniture/furniture_train_dials_target.txt", 'r')
w = open("./../data/furniture_sampled/furniture_train_dials_target.txt", 'w')

BF = " => Belief State : "
EOB = "<EOB>"

def sample_data(reader, writer):
    for line in reader.readlines():
        sp = line.strip().split(BF)
        belief_state = sp[1]
        belief_state = belief_state.split(EOB)
        state = belief_state[0]
        response = belief_state[1]
        

