 
from langchain_openai import OpenAI, ChatOpenAI  
import os
# # Text Completion (기존 llm)

chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

content = "content"



# print("Chat 응답:", result2)

import streamlit as st

st.title("Our playground")

content = st.text_input('Ask me anything.')
if st.button('Answer me!'):
    with st.spinner('Wait...'):

        result = chat_model.predict("Tell me about the" + content)
    
        st.write(result)
             

#st.write('시의 주제는', title)
st.write("---")
from datetime import datetime

st.title("💬 댓글 남기기 예제")

# 세션 상태에 댓글 리스트 없으면 초기화
if "comments" not in st.session_state:
    st.session_state.comments = []

# 입력 폼
with st.form("comment_form", clear_on_submit=True):
    username = st.text_input("이름을 입력하세요")
    comment = st.text_area("댓글을 남겨주세요")
    submitted = st.form_submit_button("등록")

    if submitted and username and comment:
        st.session_state.comments.append({
            "user": username,
            "text": comment,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# 댓글 출력
st.write("### 📜 댓글 목록")
if st.session_state.comments:
    for c in reversed(st.session_state.comments):
        st.markdown(f"**{c['user']}** · *{c['time']}*")
        st.write(c['text'])
        st.divider()
else:
    st.info("아직 댓글이 없습니다. 첫 댓글을 남겨보세요!")





