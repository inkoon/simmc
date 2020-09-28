import json
import argparse

# Step1 : get tokenlist

parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=str,
        help="choose domain {furniture|fashion}")
args = parser.parse_args()

domain = args.domain

f = open(f"gpt2_dst/data/{domain}/special_tokens.json", 'r')
act_w = open(f"gpt2_dst/data/{domain}/act.json",'w')
slot_w = open(f"gpt2_dst/data/{domain}/slot.json",'w')

tokens = json.load(f)
da_set = set()
err_set = set()
slot_set = set()

for tok in tokens["additional_special_tokens"]:
    if tok.startswith("DA"):
        da_set.add(tok)
    elif tok.startswith("ERR"):
        err_set.add(tok)
    elif tok.startswith("<"):
        continue
    else :
        slot_set.add(tok)

da = list(da_set)
err = list(err_set)
slot = list(slot_set)
da.sort()
err.sort()
slot.sort()

token_list = []

for item in da:
    token_list.append(item)
for item in err:
    token_list.append(item)
for item in slot:
    token_list.append(item)

da.extend(err)

act_json = json.dumps(da)
slot_json = json.dumps(slot)
act_w.write(act_json)
slot_w.write(slot_json)

# Step2 : make dict

token_to_special = open(f"gpt2_dst/data/{domain}/token_to_special.json",'w')
special_to_token = open(f"gpt2_dst/data/{domain}/special_to_token.json",'w')


token_match = dict()
match_token = dict()

for line in token_list:
    token = line.strip()
    if '.' in token:
        sp = token.split('.')
        new1 = sp[0].lower()
        new1 = new1.replace(':', ' ')
        new1 = new1.replace('_', ' ')
        
        new2 = ""
        for c in sp[1]:
            if c.isupper():
                new2+=' '+c.lower()
            else:
                new2+=c
        
        new_token = new1 + ' ' + new2
        token_match[new_token+' '] = token+ ' '
        match_token[token+ ' '] = new_token+ ' '

    elif ':' in token:
        new_token = token.lower().replace(':', ' ').replace('_', ' ')
        token_match[new_token+' '] = token+ ' '
        match_token[token+ ' '] = new_token+ ' '

    elif '-' in token:
        sp = token.split('-')
        new = ""
        if sp[1][0].isupper(): # if not a camelCased token (eg: NEG)
            new = sp[1]
        else:
            if '_' in sp[1]:
                sp[1] = sp[1].replace('_', ' ')
            for c in sp[1]:
                if c.isupper():
                    new+=' ' + c.lower()
                else:
                    new+=c
        new_token = sp[0] + ' ' + new

        token_match[new_token+' '] = token+ ' '
        match_token[token+ ' '] = new_token+ ' '

json_dump  = json.dumps(token_match)
json_dump2 = json.dumps(match_token)
token_to_special.write(json_dump)
special_to_token.write(json_dump2)
