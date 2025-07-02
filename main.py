import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)
OpenAiBaseUrl = os.getenv("OPENAI_BASE_URL")


def chat(_message, _model):
    result = _model.invoke(_message)
    print(result.content.strip())


def stream(_message, _model):
    result = _model.stream(_message)
    for chunk in result:
        print(chunk.content, end="", flush=True)


def basic_chat_chain(_message, _prompt_template, _model):
    chain = _prompt_template | _model | StrOutputParser()
    result = chain.invoke({"question": _message})
    print(result)


def print_separator():
    for i in range(200):
        print("-", end="")
        print()


if __name__ == "__main__":
    prompt_template = ChatPromptTemplate([
        ("system", "You are a helpful assistant, Please provide answers based on the user's questions."),
        ("human", "hi, {question}"),
    ])
    model = init_chat_model(model="gemini-2.5-flash-lite-preview-06-17", model_provider="openai", base_url=OpenAiBaseUrl)

    question = "hi, tell me about you?"

    chat(question, model)
    print_separator()
    stream(question, model)
    print_separator()
    basic_chat_chain(question, prompt_template, model)
