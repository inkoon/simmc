import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

predicted = open(args.path + 'furniture_devtest_dials_predicted.txt', 'r')

predicted_processed = open(args.path + "furniture_devtest_dials_predicted_processed.txt", 'w')

act_path = open("gpt2_dst/data/act.json", 'r')

act_list = json.load(act_path)

BELIEF_STATE = " => Belief State : "
EOB = " <EOB> "


def postprocess(reader, writer):
    for line in reader.readlines():
        split = line.split(BELIEF_STATE)
        prompt = split[0]
        bs = split[1]
        
        split = bs.split(EOB)
        state = split[0]
        response = split[1]

        writer.write(prompt)
        writer.write(BELIEF_STATE)
        
        state_split = state.split(' ')
        while '' in state_split:
            state_split.remove('')
        for i, token in enumerate(state_split):
            if i == 0:
                continue
            if i == (len(state_split)):
                break
            if state_split[i-1] in act_list and state_split[i+1] == '[':
                writer.write(state_split[i-1]+'.')
            else:
                writer.write(state_split[i-1]+' ')
        writer.write(state_split[len(state_split)-1])
        
        writer.write(EOB)
        writer.write(response)

# postprocess(target, target_processed)
postprocess(predicted, predicted_processed)



