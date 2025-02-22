

import os
import requests
import json
import yfinance as yf
from openai import OpenAI
import streamlit as st
import matplotlib.pyplot as plt
import re
import pandas as pd


import openai
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
import chromadb 
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase



client=OpenAI(api_key=os.environ["OPENAI_API_KEY"])
serpapi=os.environ["SERP_API_KEY"]
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY




def topicclassifier(input):
    topic_template="""You get input from user and find it is for searching news or serching for stock data or
    doing other topic.\

    if input's topic is about news say just "news",
    if input's topic is about stock price or finantial data say just "stock price".
    if input's topic is about this pdf file say just "enviromental_economy"
    This is the summary of pdf file.
    "Summary of the Environmental Economics PDF
    The document primarily discusses the theory and policy aspects of environmental economics, focusing on the internalization of externalities as a central theme. Here are the key points covered:

    Foundations of Microeconomic Theory:

    Explains scarcity and resource allocation.
    Discusses the broad concepts of needs and resources, including both traditional economic goods (food, shelter) and extra-economic needs (clean environment, security).
    Allocation Mechanisms:

    Different mechanisms for distributing scarce resources, such as market mechanisms, democratic processes, and authoritarian allocations.
    Example: Air pollution and the competing claims of firms and residents on clean air.
    Positive and Normative Analysis:

    Positive analysis examines how individual decisions aggregate in an economy.
    Normative analysis evaluates these outcomes against social welfare criteria.
    Market Equilibrium and Social Optimality:

    Discusses market equilibrium where supply and demand intersect.
    Social optimality involves maximizing aggregate welfare, which includes both benefits and costs.
    Externalities and Market Failure:

    Externalities occur when a decision impacts others who are not involved in the transaction, leading to market failure.
    Examples: Pollution affecting non-consenting parties.
    Internalization of Externalities:

    Strategies to internalize external costs, such as taxes, regulations, and market-based approaches.
    The goal is to align private incentives with social welfare.
    Environmental Policy Implications:

    Discusses various policy instruments to address environmental issues, including Pigovian taxes, cap-and-trade systems, and liability rules.
    Economic Models and Social Welfare:

    Use of theoretical models to understand environmental economics.
    The importance of assumptions in model building and their implications for real-world policy.
    Principles of Consumer Sovereignty and Utility Measurement:

    Emphasizes that consumers preferences should guide resource allocation.
    Challenges in measuring utility and translating it into policy decisions.

    Case Studies and Applications:
    Provides examples and applications of the discussed theories in real-world scenarios."

    If the query is about asking statiscal topic about montly green house gas emission data from 2018 to 2024, respond with "statistic"

    If the query is about any other topic, respond with "something else".

    Query: {query}
    """
    topic_prompt=PromptTemplate(input_variables = ['query'],template = topic_template)
    llm = ChatOpenAI(model="gpt-4-turbo")
    llm_chain = topic_prompt | llm | StrOutputParser()
    response=llm_chain.invoke({"query": input})
    return response

def google_news(topic):
    params={
        "engin":"google",
        "tbm":"nws",
        "q":topic,
        "api_key":os.environ["SERP_API_KEY"],
    }
    response=requests.get('https://serpapi.com/search', params=params)
    data=response.json()
    return data.get('news_results')

def get_news(topic):
    news_ = google_news(topic)  # Assuming this function gets news data for the ticker
    news_list = []

    for news in news_:
        if news is not None:
            title = news.get("title", "No title")
            link = news.get('link', 'No link')
            source = news.get("source", "No source")
            day = re.findall(r'\d+', news['date'])[0]
            day = int(day)
            date = (datetime.today() - pd.Timedelta(f"{day}d")).strftime("%Y-%m-%d")

            news_item = {
                "title": title,
                "link": link,
                "source": source,
                "date": date
            }
            news_list.append(news_item)

    return news_list

def get_financial_statements(ticker):
    company = yf.Ticker(ticker)
    data = {
        "balance_sheet": None,
        "cash_flow": None,
        "income_statement": None,
        "valuation_measures": None
    }

    try:
        balance_sheet = company.balance_sheet
        if isinstance(balance_sheet, pd.DataFrame):
            balance_sheet.index = balance_sheet.index.astype(str)
            balance_sheet.columns = balance_sheet.columns.astype(str)
            data["balance_sheet"] = balance_sheet.to_dict(orient='index')
    except Exception as e:
        print(f"Error fetching balance sheet: {e}")
        data["balance_sheet"] = None

    try:
        cash_flow = company.cashflow
        if isinstance(cash_flow, pd.DataFrame):
            cash_flow.index = cash_flow.index.astype(str)
            cash_flow.columns = cash_flow.columns.astype(str)
            data["cash_flow"] = cash_flow.to_dict(orient='index')
    except Exception as e:
        print(f"Error fetching cash flow: {e}")
        data["cash_flow"] = None

    try:
        income_statement = company.financials
        if isinstance(income_statement, pd.DataFrame):
            income_statement.index = income_statement.index.astype(str)
            income_statement.columns = income_statement.columns.astype(str)
            data["income_statement"] = income_statement.to_dict(orient='index')
    except Exception as e:
        print(f"Error fetching income statement: {e}")
        data["income_statement"] = None

    try:
        valuation_measures = company.info
        # Convert any non-serializable values to strings
        for key, value in valuation_measures.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                valuation_measures[key] = str(value)
        data["valuation_measures"] = valuation_measures if isinstance(valuation_measures, dict) else None
    except Exception as e:
        print(f"Error fetching valuation measures: {e}")
        data["valuation_measures"] = None
    
    return json.dumps(data, indent=4)

def get_stock_evolution(company_name):
    company = yf.Ticker(company_name)
    hist = company.history(period="1y")
    hist.reset_index(inplace=True)
    return hist

def get_data(company_name, period="1y"):
    hist = get_stock_evolution(company_name)
    data = get_financial_statements(company_name)
    return hist, data


def get_news_agent(request):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{
            "role": "user",
            "content": f"Given the user request, make the query to find news about that and find news for topic using query you make: {request}?"
        }],
        functions =[
            {'name': 'get_news',
            'description': 'get news for ticker interest and save as data frame',
            'parameters':{
                "type": "object", 
                "properties":{
                    "topic":{
                        "type": "string", 
                        "description": "query for search news",
                    }
                },
            "required": ["topic"]}
            }
        ],
        function_call={"name": "get_news"}
    )

    message = response.choices[0].message

    if message.function_call:
        arguments = json.loads(message.function_call.arguments)
        topic = arguments["topic"]
        news=get_news(topic)
        news_json = json.dumps({"news_list": news})
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": request
                },
                message,
                {
                    "role": "system",
                    "content": """List the article title, publication date, and URL of the given news content in the format below. User
                    If there is news with a similar topic, group the news by topic.

                    Topic: similar topic of the news

                    Title: title of the news
                    Date: date of the news
                    URL: url of the news
                    Summary: breif summary of news contents

                    Title: title of the news
                    Date: date of the news
                    URL: url of the news
                    Summary: breif summary of news contents
                    """
                },
                {
                    "role": "assistant",
                    "content": news_json
                },
            ],
        )

        return second_response.choices[0].message.content
    else:
        return message
    
def get_openai_embeddings(text, model="text-embedding-3-large"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "input": text,
        "model": model
    }
    response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
    response_data = response.json()
    embedding = response_data["data"][0]["embedding"]
    return embedding

def get_information(query):
    model_name = "text-embedding-3-large"  
    persist_directory = "./chroma_db"

    client = chromadb.PersistentClient(path=persist_directory)
    vectorstore = client.get_collection(name='environment')  

    query_embedding = get_openai_embeddings(query, model=model_name)

    results = vectorstore.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    results_json = json.dumps(results)
    return results_json


def get_statical_info(input):
    db_path="C:/Users/fpqhs/chatgpto/environment.db"
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    agent = create_sql_agent(llm, db=db, agent_type="openai-tools")
    message=agent.invoke(input)
    message_str = json.dumps(message)

    message_dict = json.loads(message_str)
    output_str = message_dict["output"]
    return output_str
    
    

def get_environment_economy_information_agent(request):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{
            "role": "user",
            "content": f"Given the user request, make an appropriate question for get what user want to get and find information about it using get_information functio:{request}?"
        }],
        functions =[
            {'name': 'get_informantion',
            'description': 'function to get answer about enviromental economic question using enviromental economy book',
            'parameters':{
                "type": "object", 
                "properties":{
                    "query":{
                        "type": "string", 
                        "description": "query for search",
                    }
                },
            "required": ["query"]}
            }
        ],
        function_call={"name": "get_informantion"}
    )

    message = response.choices[0].message

    if message.function_call:
        arguments = json.loads(message.function_call.arguments)
        query = arguments["query"]
        result=get_information(query)
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": request
                },
                message,
                {
                    "role": "system",
                    "content": """
                    You are agent for answering what user asked by using result searched 
                    by using vector space. 
                    Checks whether the information related to the user's request is correct 
                    and uses the information to generate an answer to the user's request.
                    If you using result(content), just using content and not adding any information beside content.
                    """
                },
                {
                    "role": "assistant",
                    "content": result
                },
            ],
        )

        return second_response.choices[0].message.content
    else:
        return message

   

def get_stockprice_agent(request):
    response=client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{
            "role":
            "user",
            "content": f"Given the user request, convert company name to company ticker and get financial data of company using company ticker:{request}?"
        }],
        functions=[{
            "name": "get_data",
            "description": "Function to get graph of stock price of company and financial data of company",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The ticker of the company"
                    },
                    "period": {
                        "type": "string",
                        "description": "The period of analysis"
                    },
                },
                "required": ["company_name"]
            }
        }],
        function_call={"name": "get_data"}
    )#chat gpt한테 function 주고 message 줘서 응답받기

    message=response.choices[0].message

    if message.function_call:
        #Parse the return value from a JSON string to a Python dictionary
        arguments=json.loads(message.function_call.arguments)
        company_name=arguments["company_name"]

        hist, data=get_data(company_name)

        
        second_response=client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role":"user",
                "content": request
                },
                message,#ai한테 미리 주의사항을 지시 해줌
                {
                    "role": "system", 
                    "content": 
                    """analyze stock price, flow, valuation_measures, and balance sheet about company using content.\
                    And make information easy to understand.\ using paragraph and devide information about stock price, flow, valuation_measures, and balance sheet.\
                    Just using content and not adding any information beside content
                    """
                },
                {
                    "role":"assistant",
                    "content":data,
                },
            ],
        )

        return second_response.choices[0].message.content, hist
    
def normal_agent(input):
    normal_template = """
    You are an agent who answers questions that do not correspond to the function of all other agents in the environment chatbot. 
    Never answer information you don't know.
    Here is a passage or question you need to find answer:
    {input}
    """
    
    topic_prompt = PromptTemplate(input_variables=['input'], template=normal_template)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    llm_chain = topic_prompt | llm | StrOutputParser()
    
    response = llm_chain.invoke({"input": input})
    
    return response


    

def main():
    st.title("Eco chatbot")
    input=st.text_input("Ask: ")
    enter_button=st.button("Enter")
    
    if enter_button:
        if input:
            st.write("Answering....please wait")
            topic=topicclassifier(input)
            if(topic=='news'):
                st.write("Hi, I am the News Chat Bot. I will provide you with the latest news.")
                message=get_news_agent(input)
                st.markdown(message, unsafe_allow_html=True)
            elif(topic=='stock price'):
                st.write("Hi, I am the Stock Price Chat Bot. I will provide you with stock prices and financial data.")
                investment_research, hist=get_stockprice_agent(input)
                hist_selected=hist[['Open', "Close"]]

                #create figure
                fig, ax=plt.subplots()
                #plot the data
                hist_selected.plot(kind="line", ax=ax)

                #set title and labels
                ax.set_title(f"Stock price")
                ax.set_xlabel("Date")
                ax.set_ylabel("Stock price")

                #display the plot
                st.pyplot(fig)
                st.markdown(investment_research, unsafe_allow_html=True)
            elif(topic=='enviromental_economy'):
                st.write("Hi, I am the Environmental Economics Chat Bot. I will provide you with information related to environmental economics.")
                message=get_environment_economy_information_agent(input)
                st.write(message)
            elif(topic=="statistic"):
                st.write("Hi, I am the Statiscal Information Chat Bot. I know montly Co2 emission data from 2018 to 2024")
                message=get_statical_info(input)
                st.write(message)
            else:
                st.write("Hi, I am the General Information Chat Bot. I am here to assist you with any other questions you may have.")
                message=normal_agent(input)
                st.write(message)
                


if __name__=="__main__":
    main()