import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()

f = open(args.path, 'r')
w = open(args.path+'.eob', 'w')

for line in f.readlines():
    w.write(line.strip() + " <EOB>\n")


