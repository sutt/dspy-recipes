import dspy

class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

def simple_predict(question: str) -> str:

    generate_answer = dspy.Predict(BasicQA)
    generate_answer()