import os, dotenv, subprocess
dotenv.load_dotenv()
if os.environ.get('OPENAI_API_KEY') is None:
    print("couldn't find OPENAI_API_KEY in .env file")
else:
    OPENAI_API_KEY  = os.environ['OPENAI_API_KEY']
    print(OPENAI_API_KEY)