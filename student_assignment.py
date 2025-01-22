import json
import traceback
import requests

from model_configurations import get_model_configuration

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain import hub
from pydantic import BaseModel, Field
from langchain_core.utils.json import parse_json_markdown
import base64
from mimetypes import guess_type


gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

calenderific_api_key = 'XD2n3twlW3RxUJtTOBfqx2E4TdvXVMxd'

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

def get_formatted_output(llm: BaseChatModel, input: str, isList: bool) -> str:
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

def get_calenderific_api_url(year: int, month: int) -> str:
    return f"https://calendarific.com/api/v2/holidays?&api_key={calenderific_api_key}&country=tw&year={year}&month={month}"

def create_calender_tool():
    def get_holidays(year: int, month: int) -> str:
        response = requests.get(get_calenderific_api_url(year, month))
        return response.json().get('response')

    class GetHolidays(BaseModel):
        year: int = Field(description="year")
        month: int = Field(description="month")

    return StructuredTool.from_function(
        name="get_holidays",
        description="Get holiday with provided year and month.",
        func=get_holidays,
        args_schema=GetHolidays,
    )

def get_calender_agent(llm: BaseChatModel) -> AgentExecutor:
    prompt = hub.pull("hwchase17/openai-functions-agent")
    tools = [create_calender_tool()]
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools)

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def generate_hw01(question: str) -> str:
    llm = get_llm();
    response = get_holiday_output(llm, question)
    response = get_formatted_output(llm, response, True)

    return response
    
def generate_hw02(question):
    llm = get_llm()
    agent_executor = get_calender_agent(llm)
    response = agent_executor.invoke({"input": question}).get('output')
    response = get_holiday_output(llm, response)
    response = get_formatted_output(llm, response, True)
    
    return json.dumps(parse_json_markdown(response), ensure_ascii=False)
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    llm = get_llm()
    image_path = 'baseball.png'
    data_url = local_image_to_data_url(image_path)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "辨識圖片中的文字表格"),
            ("user", [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            ]),
            ("human", "{question}")
        ]
    )

    return llm.invoke(prompt.format_messages(question=question)).content
    
def demo(question):
    llm = get_llm()
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

question = "中華台北的積分"
print(generate_hw04(question))