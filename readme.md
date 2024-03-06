# DSPY Exercises
3.2.24 - 3.4.24

#### setup
add openai key to env var:
> myenv.bat

## intro-book-1
#### notebooks
 - orig.ipynb: the original book intro.ipynb fro the DSPY repo
 - mimic-1.ipynb: stripping out the code cells
 - mimic-2.ipynb: focusing on building a compiled rag and exporting/importing it
#### py
 - infer.py - dspy signatures, inference object, command line
 - server.py - basic flask server
 - client.py - basic flask client for infer/ endpoint
 - opt.py - export a compiled rag to a file (to be loaded by infer.py)
#### commands:
1. Run the server and client:

```bash
python server.py
python client.py
```

1. Run simple inference:
Use the cli args in `infer.py` to get a completion without a to a commandline question server. Will default to sigtype='basicqa'

```bash
python infer -q "What is the largest city in the United States?"
```

Other ways of calling it allow loading from a file, or using a different sig_type.

```bash
python -d hotpot1 --basicrag
```

1. Fancy Inference:

Demonstrate progressive use of `sig_type` `'basicqa'`, `'basiccot'`, to get an answer correctly:

```bash
>python infer.py -q "What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?" --basicqa    
Question: What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?

# Answer: American (incorrect)

>python infer.py -q "What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?" --basiccot   
Question: What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?

# Answer: British (correct, is actually "English")
```

Demonstrate progressive use of `sig_type` `'basicqa'`, `'basiccot'`, `'basicrag'` to get an answer correctly:
```bash
python infer.py -q "Who acted in the shot film The Shore and is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of \"Hamlet.\" ?" --basicqa 

# Answer: Ciarán Hinds (incorrect)

python infer.py -q "Who acted in the shot film The Shore and is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of \"Hamlet.\" ?" --basiccot

# Answer: Saoirse Ronan (incorrect)

python infer.py -q "Who acted in the shot film The Shore and is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of \"Hamlet.\" ?" --basicrag

# Answer: Kerry Condon (correct)
```
