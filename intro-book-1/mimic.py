import dspy

import os
from dotenv import load_dotenv
load_dotenv('../.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

def main(question: str):

    turbo = dspy.OpenAI(
        model='gpt-3.5-turbo', 
        api_key=OPENAI_API_KEY,
    )
    
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(
        url='http://20.102.90.50:2017/wiki17_abstracts'
    )

    dspy.settings.configure(
        lm=turbo, 
        rm=colbertv2_wiki17_abstracts
    )

    generate_answer = dspy.Predict(BasicQA)
    pred = generate_answer(question=question)

    print(pred)

if __name__ == "__main__":
    question = "Where is Amelia Earhart from?"
    import sys
    if sys.argv[1:]:
        question = sys.argv[1]
    print(question)
    main(question)