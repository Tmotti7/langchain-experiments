from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())


def draft_sales_response(user_input, name="Dave"):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """
        
    You are a helpful salesperson offering AI and Automation products. Reply based on: you sell - slack company that builed with AI bots that can help run your buisness better.
    
    Your goal is to help the user navigate any sales objections or queries they encounter.
    
    Keep your response short, precise, and tailor it to the specific question or objection raised by the customer.
    
    Start your response by saying: "Hi {name}, to address that:". And then proceed with the response on a new line.
    
    Conclude with: {closing} to motivate the customer toward a positive decision.
    
    """

    closing = "If you have any more questions or need further clarification, I'm here to help. Let's find the best solution for your needs!"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's the customer's query or objection: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, closing=closing, name=name)

    return response
