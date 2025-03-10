from dotenv import load_dotenv
from langfuse.callback import CallbackHandler

from rcsa.summarizer.graph import app

load_dotenv()

langfuse_handler = CallbackHandler()
app = app.with_config({"callbacks": [langfuse_handler]})

if __name__ == "__main__":
    print(app.invoke({"contents": ["Today is a good day", "I am happy"]}))
