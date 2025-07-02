import string

import pandas
from langchain_core.output_parsers import JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_experimental.tools import PythonAstREPLTool

import main


def code_print(res):
    print("即将运行的代码：" + res['query'])
    return res


if __name__ == '__main__':
    prompt_template = ChatPromptTemplate([
        ("system",
         "你可以访问一个名为 `dataset` 的 pandas dataframe,你可以使用df.head().to_markdown() 查看数据集的基本信息,请根据用户提出的问题,编写 Python 代码来回答。只返回代码,不返回其他内容。只允许使用 pandas 和内置库。"),
        ("user", "{message}")
    ])

    dataset = pandas.read_csv(filepath_or_buffer='resource/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    tool = PythonAstREPLTool(locals={"dataset": dataset})
    print_node = RunnableLambda(code_print)
    tool_call_chain = prompt_template | main.model.bind_tools([tool]) | JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True) | print_node | tool

    print(tool_call_chain.invoke({"message": "请帮我计算MonthlyCharges列的均值"}))
    print(tool_call_chain.invoke({"message": "请帮我计算一共有多少行数据"}))
    print(tool_call_chain.invoke({"message": "查一下全表中不限字段的最高值"}))
