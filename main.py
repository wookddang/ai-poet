# from dotenv import load_dotenv
# load_dotenv()

from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
import streamlit as st
import time
chat_model = ChatOpenAI()
st.title('인공지능 시인')

content = title = st.text_input('시의 주제를 제시해주세요.')

if st.button("시 작성 요청하기"):
    with st.spinner("시 작성 중..."):

        result = chat_model.predict(content + "에 대한 시를 써줘")
        st.write(result)




# import streamlit as st
# st.title('This is a title')
# st.title('_Streamlit_is : blue[cool] : sunglasses:')
