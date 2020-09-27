import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--domain', type=str, required=True)
args = parser.parse_args()

predicted = open(args.path + args.domain+ '_devtest_dials_predicted.txt', 'r')
predicted_processed = open(args.path + args.domain + "_devtest_dials_predicted_response.txt", 'w')

BELIEF_STATE = " => Belief State : "
EOB = " <EOB> "

def postprocess(reader, writer):
    for i, line in enumerate(reader.readlines()):
        # writer.write(str(i) + '\t')
        split = line.split(BELIEF_STATE)
        prompt = split[0]
        bs = split[-1]
        
        split = bs.split(EOB)
        state = split[0]
        response = split[-1]
        writer.write(response)

# postprocess(target, target_processed)
postprocess(predicted, predicted_processed)


