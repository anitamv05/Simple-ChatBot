import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Define the conversation template and memory
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""
    The following is a conversation between a user and an AI assistant. The assistant is helpful, creative, clever, and very friendly.

    {history}

    User: {input}
    Assistant:"""
)

memory = ConversationBufferMemory()
llm = ChatOpenAI(model="gpt-4", temperature=0.6)
conversation = ConversationChain(llm=llm, prompt=prompt_template, memory=memory)

# Streamlit app layout
st.title("Chatbot Demo")
st.write("Ask anything and get a response from the AI!")

# User input
user_input = st.text_input("You: ", "Hello, who are you?")

# Generate response
if user_input:
    response = conversation.predict(input=user_input)
    st.write(f"AI: {response}")
