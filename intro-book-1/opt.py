import json
import dspy
from dspy.teleprompt import BootstrapFewShot

import os
from dotenv import load_dotenv
load_dotenv('../.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

turbo = dspy.OpenAI(model='gpt-3.5-turbo', api_key=OPENAI_API_KEY)
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)

train_fn = 'data/trainset.json'
with open(train_fn, 'r') as f:
    load_trainset = json.load(f)
trainset = [dspy.primitives.example.Example(**e) for e in load_trainset]
trainset = [e.with_inputs('question') for e in trainset]

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

def compile_rag() -> RAG:
    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
    compiled_rag = teleprompter.compile(RAG(), trainset=trainset)
    return compiled_rag

def export_rag(program: RAG, export_fn: str = 'data/export_1.json') -> None:
    program.save(export_fn)

def main():
    rag = compile_rag()
    out = rag(question=trainset[0].question)
    print(out)
    export_rag(rag, export_fn='data/export_1.json')

if __name__ == '__main__':
    main()