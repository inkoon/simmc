import json

f = open("tokenlist.txt", 'r')
w = open("special_token_match.json", 'w')

token_match = dict()

for line in f.readlines():
    token = line.strip()
    if '.' in token:
        sp = token.split('.')
        new = ""
        for c in sp[1]:
            if c.isupper():
                new+=' '+c.lower()
            else:
                new+=c
        new_token = sp[0] + ' ' + new
        token_match[new_token] = token
    elif '-' in token:
        sp = token.split('-')
        new = ""
        for c in sp[1]:
            if c.isupper():
                new+=' ' + c.lower()
            else:
                new+=c
        new_token = sp[0] + ' ' + new
        token_match[new_token] = token

json_dump  = json.dumps(token_match)
w.write(json_dump)
