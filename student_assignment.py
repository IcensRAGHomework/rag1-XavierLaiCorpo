import json
import traceback

from model_configurations import get_model_configuration

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

def get_holiday_output(llm: BaseChatModel, input: str) -> str:
    response_schemas = [
        ResponseSchema(name="date", description="該節日的日期", type="YYYY-MM-DD"),
        ResponseSchema(name="name", description="該節日的名稱")
    ]
    parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = parser.get_format_instructions()

    system_req = f"使用台灣習慣的繁體中文來回答問題，將我提供的資料整理成指定格式，{format_instructions}，並將所有答案放進同個list"
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "{system_req}"),
        ("human", "{input}")
    ])
    chat_template = chat_template.partial(system_req=system_req)
    return llm.invoke(chat_template.format_messages(input=input)).content

def get_formatted_output(llm: BaseChatModel, input: str) -> str:
    examples = [
        {"input": """```json
                    {
                            "Result": [
                                    content
                            ]
                    }
                    ```""",
        "output": """{
                            "Result": [
                                    content
                            ]
                    }"""},
    ]
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)
    formatted_chat_template = ChatPromptTemplate.from_messages([
        ("system", "將我提供的文字進行整理"),
        few_shot_prompt,
        ("human", "{input}")
    ])
    
    return llm.invoke(formatted_chat_template.format(input=input)).content

def generate_hw01(question: str) -> str:
    llm = get_llm();
    holiday_response = get_holiday_output(llm, question)
    formatted_response = get_formatted_output(llm, holiday_response)

    return formatted_response
    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = get_llm()
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

#question = "請回答台灣2024年10月的紀念日有哪些"
#print(generate_hw01(question))