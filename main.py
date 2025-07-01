import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)
OpenAiBaseUrl = os.getenv("OPENAI_BASE_URL")


def chat(_message, _model):
    result = _model.invoke(_message)
    print(result.content.strip())


def stream(_message, _model):
    result = _model.stream(_message)
    for chunk in result:
        print(chunk.content, end="", flush=True)


def basic_chat_chain(_message, _model):
    chain = _model | StrOutputParser()
    result = chain.invoke(_message)
    print(result)


def print_separator():
    for i in range(200):
        print("-", end="")
        print()


if __name__ == "__main__":
    message = [
        ("human", "hi, tell me about you?"),
    ]

    model = init_chat_model(model="gemini-2.5-flash-lite-preview-06-17", model_provider="openai", base_url=OpenAiBaseUrl)

    chat(message, model)
    print_separator()
    stream(message, model)
    print_separator()
    basic_chat_chain(message, model)
