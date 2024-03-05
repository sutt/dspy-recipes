import requests

base_url = 'http://localhost:5000/'
end_point = 'infer'

def send_infer(
        question: str,
    ) -> str:
    data = {
        'question': question,
        'sig_type': 'BasicQA',
    }
    response = requests.post(base_url + end_point, json=data)
    if response.status_code == 200:
        result = response.json()
        return result.get('answer')

if __name__ == "__main__":
    question = "Where is Amelia Earhart from?"
    answer = send_infer(question)
    print(answer)
