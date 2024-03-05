import json
'''
    convert a text file to a json file
    one-line per question
'''
fn = 'hotpot1.txt'

with open(fn, 'r') as f:
    questions_raw = f.readlines()

questions = [q.strip() for q in questions_raw]

question_obj = {i: q for i, q in enumerate(questions)}

out_fn = fn.replace('.txt', '.json')
with open(out_fn, 'w') as f:
    f.write(json.dumps(question_obj, indent=2))