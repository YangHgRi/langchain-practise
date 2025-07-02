import json
import os

import pandas
import requests
from langchain_core.output_parsers import JsonOutputKeyToolsParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_experimental.tools import PythonAstREPLTool

import main


def code_print(res):
    print("即将运行的代码：" + res['query'])
    return res


@tool
def get_weather(loc: str) -> str:
    """
    查询即时天气函数
    :param loc: 必要参数,字符串类型,用于表示查询天气的具体城市名称,\
    注意,中国的城市需要用对应城市的英文名称代替,例如如果需要查询北京市天气,则loc参数需要输入'Beijing'.
    :return: Openweather API查询即时天气的结果,\
     返回结果对象类型为解析之后的JSON格式对象,并用字符串形式进行表示,其中包含了全部重要的天气信息.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q":     loc,
        "appid": os.getenv("OPENWEATHER_API_KEY"),
        "units": "metric",
        "lang":  "zh_cn",
    }

    response = requests.get(base_url, params=params)
    data = response.json()
    return json.dumps(data)


def use_build_in_tool(user_message: str) -> str:
    prompt_template = ChatPromptTemplate([
        ("system",
         "你可以访问一个名为 `dataset` 的 pandas dataframe,你可以使用df.head().to_markdown() 查看数据集的基本信息,请根据用户提出的问题,编写 Python 代码来回答。只返回代码,不返回其他内容。只允许使用 pandas 和内置库。"),
        ("user", "{message}")
    ])
    dataset = pandas.read_csv(filepath_or_buffer='resource/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    _tool = PythonAstREPLTool(locals={"dataset": dataset})
    print_node = RunnableLambda(code_print)
    tool_call_chain = prompt_template | main.model.bind_tools([_tool]) | JsonOutputKeyToolsParser(key_name=_tool.name, first_tool_only=True) | print_node | _tool

    return tool_call_chain.invoke({"message": user_message})


def use_custom_tool(user_message: str) -> str:
    tool_calling_prompt_template = ChatPromptTemplate([
        ("system",
         "你可以访问一个名为 `get_weather` 的外部工具函数查看实时天气信息,请根据用户提出的问题,返回函数参数,不返回其他内容。"),
        ("user", "{message}")
    ])
    output_conversion_prompt_template = PromptTemplate.from_template(
        """
        你将收到一段 JSON 格式的天气数据,请用简洁自然的方式将其转述给用户。
        以下是天气 JSON 数据:
        ```json
        {weather_json}
        ```
        请将其转换为全面但简洁的中文天气描述
        只返回一句话描述, 不要其他说明或解释。
        """
    )

    _tool = get_weather
    tool_call_chain = tool_calling_prompt_template | main.model.bind_tools([_tool]) | JsonOutputKeyToolsParser(key_name=_tool.name, first_tool_only=True) | _tool
    format_output_for_prompt = RunnableLambda(lambda weather_json_str: {"weather_json": weather_json_str})
    output_chain = output_conversion_prompt_template | main.model | StrOutputParser()
    full_chain = tool_call_chain | format_output_for_prompt | output_chain

    return full_chain.invoke({"message": user_message})


if __name__ == '__main__':
    print(use_build_in_tool("请帮我计算一共有多少行数据"))
    print(use_custom_tool("现在锦州市的天气如何？"))
