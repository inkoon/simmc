import json

f = open("tokenlist.txt", 'r')
w = open("token_to_special.json", 'w')
w1 = open("special_to_token.json", 'w')

token_match = dict()
match_token = dict()

for line in f.readlines():
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
        token_match[new_token] = token
        match_token[token] = new_token
    elif '-' in token:
        sp = token.split('-')
        new = ""
        if sp[1] == 'DISPLAY_1':
            new = 'DISPLAY_1'
        elif sp[1] == 'NEG':
            new = 'NEG'
        for c in sp[1]:
            if c.isupper():
                new+=' ' + c.lower()
            else:
                new+=c
        new_token = sp[0] + ' ' + new
        token_match[new_token] = token
        match_token[token] = new_token

json_dump  = json.dumps(token_match)
json_dump2 = json.dumps(match_token)
w.write(json_dump)
w1.write(json_dump2)
