import dspy

import os
from dotenv import load_dotenv
load_dotenv('../.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class BasicRAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
    

class InferObj:
    def __init__(self, **kwargs):
        self.setup_client_objs(**kwargs)
    def setup_client_objs(self, **kwargs):
        self.turbo = dspy.OpenAI(
            model='gpt-3.5-turbo', 
            api_key=OPENAI_API_KEY,
        )
        self.colbertv2_wiki17_abstracts = dspy.ColBERTv2(
            url='http://20.102.90.50:2017/wiki17_abstracts'
        )
        dspy.settings.configure(
            lm=self.turbo, 
            rm=self.colbertv2_wiki17_abstracts
        )
    def generate_pred(
            self, 
            question: str,
            sig_type: str,
        ) -> dspy.predict.Predict:
        if sig_type == 'BasicQA':
            generate_answer = dspy.Predict(BasicQA)
            pred = generate_answer(question=question)
        elif sig_type == 'BasicCOT':
            generate_cot = dspy.ChainOfThought(BasicQA)
            pred = generate_cot(question=question)
        elif sig_type == 'BasicRAG':
            generate_answer = BasicRAG()
            pred = generate_answer(question=question)
        else:
            print(f"Signature type {sig_type} not recognized")
            pred = None
        return pred
    def generate_answer(
            self, 
            question: str,
            sig_type: str = 'BasicQA',
        ) -> str:
        pred = self.generate_pred(question, sig_type=sig_type)
        
        if hasattr(pred, 'answer'):
            answer = pred.answer
        else:
            try: answer = pred.completions[0].answer
            except: print(pred)
            return None
        return answer


if __name__ == "__main__":
    
    import os
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', type=str)
    parser.add_argument('-d', '--data', type=str)
    parser.add_argument('-1', '--basicqa', action='store_true')
    parser.add_argument('-2', '--basiccot', action='store_true')
    parser.add_argument('-3', '--basicrag', action='store_true')
    args = parser.parse_args()
    args = vars(args)

    # define the signature type
    if args.get('basicqa'):
        sig_type = 'BasicQA'
    elif args.get('basiccot'):  
        sig_type = 'BasicCOT'
    elif args.get('basicrag'):
        sig_type = 'BasicRAG'
    else:
        sig_type = 'BasicQA'

    # chose what data input to use
    if args.get('question'):
        questions = [args['question']]
    elif args.get('data'):
        fn = args['data']
        fp =  os.path.join('data', fn + '.json')
        if not(os.path.exists(fp)):
            raise(f"File {fn} does not exist")
        with open(fp, 'r') as f:
            questions_dict = json.load(f)
        questions = list(questions_dict.values())
    else:
        questions = ["Where is Amelia Earhart from?"]
    
    # Run inference on the question(s)
    infer_obj = InferObj()
    for question in questions:
        
        print(f"Question: {question}")
        answer = infer_obj.generate_answer(
            question, 
            sig_type=sig_type
        )
        print(f"Answer: {answer}")