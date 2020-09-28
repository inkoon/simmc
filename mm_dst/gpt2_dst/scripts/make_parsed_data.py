import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--domain', help='domain to run data parsing')
args = parser.parse_args()

domain = args.domain

f_train = open(f"gpt2_dst/data/{domain}_original/{domain}_train_dials_target.txt", 'r')
f_dev = open(f"gpt2_dst/data/{domain}_original/{domain}_dev_dials_target.txt", 'r')
# f_devtest = open(f"./{domain}/{domain2}_devtest_dials_target.txt", 'r')

w_train = open(f"gpt2_dst/data/{domain}/{domain}_train_dials_target.txt", 'w')
w_dev = open(f"gpt2_dst/data/{domain}/{domain}_dev_dials_target.txt", 'w')
# w_devtest = open(f"./{domain}_total/{domain2}_devtest_dials_target.txt", 'w')

special_to_token = open(f"gpt2_dst/data/{domain}/special_to_token.json", 'r')
special_to_token = json.load(special_to_token)

def convert_data(reader, writer):
    for line in reader.readlines():
        line = line.split(" => Belief State : ")

        for key in special_to_token.keys():
            if key in line[1]:
                line[1] = line[1].replace(key, special_to_token[key])
        writer.write(line[0]+" => Belief State : " + line[1]) 

convert_data(f_train, w_train)
convert_data(f_dev, w_dev)
# convert_data(f_devtest, w_devtest)
