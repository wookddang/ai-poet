
from langchain_openai import OpenAI, ChatOpenAI  
import os
# # Text Completion (기존 llm)

chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

content = "주제"



# print("Chat 응답:", result2)

import streamlit as st

st.title('Ola's playground')

content = st.text_input('Ask me anything.')
if st.button('Answer me!'):
    with st.spinner('Wait...'):

        result = chat_model.predict(content + "에 대한 시를 써줘")
    
        st.write(result)
             

#st.write('시의 주제는', title)


