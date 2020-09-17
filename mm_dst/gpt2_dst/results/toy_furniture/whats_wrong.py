import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--keyword", type=str, required=True)

args = parser.parse_args()

pred = open('furniture_devtest_dials_predicted_'+args.keyword+'.txt', 'r')
target = open('../../data/toy_furniture/furniture_devtest_dials_target.txt', 'r')
output = open(args.keyword + '_wrong.txt','w')

cnt=0

for i, (p, t) in enumerate(zip(pred.readlines(), target.readlines())):
    p = p.split('Belief State : ')[-1]
    t = t.split('Belief State : ')[-1]
    p = p.strip()
    t = t.strip()
    p = p.split('<EOB>')[0]+'\n'
    t = t.split('<EOB>')[0]+'\n'
    if p == t:
        output.write("")
        cnt = cnt + 1
    else:
        output.write(f'{i}\npred : {p}target :{t}')

print(cnt)

