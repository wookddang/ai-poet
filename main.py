 
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
import sqlite3
from datetime import datetime

# --- Database setup ---
conn = sqlite3.connect("comments.db")
conn.execute("""
CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    text TEXT,
    time TEXT
)
""")

# --- Page title ---
st.title("💬 Comment session")

# --- Comment input form ---
st.subheader("Leave a Comment")

with st.form("comment_form", clear_on_submit=True):
    username = st.text_input("Please enter a name.")
    comment = st.text_area("Please leave a comment.")
    submitted = st.form_submit_button("Register")

    if submitted:
        if username and comment:
            conn.execute(
                "INSERT INTO comments (user, text, time) VALUES (?, ?, ?)",
                (username, comment, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            st.success("✅ Your comment has been added successfully!")
        else:
            st.warning("⚠️ Please fill in both fields before submitting.")

# --- Display comments ---
st.subheader("📝 Comments")

rows = conn.execute("SELECT user, text, time FROM comments ORDER BY id DESC").fetchall()

if rows:
    for r in rows:
        st.markdown(f"**{r[0]}** · *{r[2]}*")
        st.write(r[1])
        st.divider()
else:
    st.info("There are no comments yet. Be the first to leave one!")






