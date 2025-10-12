import json
import os
import re
import typing as t

import openai
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

print(load_dotenv())


def read_template(template_file_name: str, params: t.Optional[dict] = None):
    current_script_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_script_path)
    with open(os.path.join(current_dir, 'message_templates', f'{template_file_name}.tpl'), 'r') as f:
        msg_template = f.read()
    return msg_template

openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

openai_template = read_template('openai_event_preparing')

chats_db = {}
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(openai_template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)


def get_json(input_str) -> dict:
    json_pattern = re.compile(r"\{.*?\}", re.DOTALL)
    json_matches = json_pattern.findall(input_str)
    res = None
    if len(json_matches) > 0:
        res = json.loads(json_matches[0])
    return res

def get_or_create_chat(user_id):
    if user_id not in chats_db:
        print('saving chat for', user_id)
        chats_db[user_id] = {
            'agent': ConversationChain(
                llm=llm,
                # chain_type="stuff",
                memory=ConversationBufferMemory(return_messages=True),
                # combine_docs_chain_kwargs={"prompt": prompt},
                verbose=True,
                prompt=prompt,
            ),
            'messages': []
        }
    return chats_db[user_id]

def dialog_router(human_input, user: dict):
    chat = get_or_create_chat(user["user_id"])
    llm_answer = chat['agent'].predict(input=human_input)
    json_response = get_json(llm_answer) # check if llm prepared final answer with structured user information
    if json_response is not None:
        return {"final_answer": True, "type": "json", "answer": json_response}
    else:
        return {"final_answer": False, "type": "text", "answer": llm_answer}

if __name__=='__main__':
    human_input = input("Start the dialog with AI travel assistant: ")
    user = {'user_id': 999}
    for k in range(10):
        answer = dialog_router(human_input=human_input, user=user)
        if answer['final_answer']:
            print(answer['answer'])
            break
        else:
            print(answer['answer'])
            human_input = input("Enter your response: ")
            print('\n..............\n')
    if k == 6:
        print("Conversation length overflow")
