from flask import Flask, request, jsonify
from infer import InferObj

app = Flask(__name__)

infer_obj = InferObj()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/process_json', methods=['POST'])
def process_json():
    data = request.get_json()
    # Process the received JSON data
    response = {
        'message': 'JSON data received and processed successfully',
        'data': data
    }
    return jsonify(response)

@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()
    print(data['question'])
    answer = infer_obj.generate_answer(data['question'], sig_type='BasicQA')
    response = {
        'answer': answer,
    }
    print(response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True,)