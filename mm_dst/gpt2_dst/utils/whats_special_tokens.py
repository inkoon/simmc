import json

domain = 'furniture'

f = open(f"../data/{domain}/special_tokens.json", 'r')
w = open(f"./{domain}/tokenlist.txt",'w')
# w2 = open(f"../data/{domain}/special_tokens2.json", 'w')

act_w = open(f"./{domain}/act.json",'w')
slot_w = open(f"./{domain}/slot.json",'w')

tokens = json.load(f)

da = []
err = []
slot = []

da_set = set()
err_set = set()
slot_set = set()

for tok in tokens["additional_special_tokens"]:
    if tok.startswith("DA"):
        da_set.add(tok)
    elif tok.startswith("ERR"):
        err_set.add(tok)
    else:
        slot_set.add(tok)

da = list(da_set)
err = list(err_set)
slot = list(slot_set)
da.sort()
err.sort()
slot.sort()

# w.write("\nDA\n")
for item in da:
    w.write(item+'\n')
# w.write("\nERR\n")
for item in err:
    w.write(item+'\n')
# w.write("\nSlots\n")
for item in slot:
    w.write(item+'\n')

da.extend(err)

act_json = json.dumps(da)
slot_json = json.dumps(slot)
act_w.write(act_json)
slot_w.write(slot_json)

# new_token_set = set()

# for d in da:
    # if '.' in d:
        # d_split = d.split('.')
        # first = d_split[0]
        # new_token_set.add(first)
        # new_token_set.add('.'+d_split[1])
    # else:
        # new_token_set.add(d)

# for item in err:
    # new_token_set.add(item)

# for item in slot:
    # new_token_set.add(item)

# tokens['additional_special_tokens'] = list(new_token_set)

# json_string = json.dumps(tokens)
# w2.write(json_string)

