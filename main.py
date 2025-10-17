import os
import re
import tempfile
from datetime import datetime
import streamlit as st
import psycopg2
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import json
from langchain.schema import HumanMessage, AIMessage

# --- Load .env ---
load_dotenv()

# --- PostgreSQL 연결 ---
conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    database=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    port=os.getenv("POSTGRES_PORT"),
    sslmode="require"
)
cur = conn.cursor()

# --- 테이블 생성 ---
cur.execute("""
CREATE TABLE IF NOT EXISTS comments (
    id SERIAL PRIMARY KEY,
    username TEXT,
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    username TEXT,
    mode TEXT,
    question TEXT,
    answer TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS embeddings_meta (
    id SERIAL PRIMARY KEY,
    username TEXT,
    filename TEXT,
    chroma_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS user_memory (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE,
    memory_text TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

conn.commit()

# --- KakaoTalk Parser ---
def KakaoTalkParser(text):
    """
    카카오톡 txt 파일에서 날짜, 발화자, 메시지를 추출
    """
    pattern = re.compile(r"(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.\s*[오전|오후]+\s*\d{1,2}:\d{2}),\s*(.*)\s*:\s*(.*)")
    data = []

    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            dt_str, sender, content = match.groups()
            try:
                date_obj = datetime.strptime(dt_str.replace("오전", "AM").replace("오후", "PM"), "%Y. %m. %d. %p %I:%M")
            except:
                continue
            data.append((date_obj, sender.strip(), content.strip()))
    return data


# --- Streamlit Layout ---
st.set_page_config(page_title="🐶 Maltipoo Chat Playground", page_icon="🐶", layout="centered")
st.title("🐾 My Maltipoo's Playground")
st.write("---")

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 PDF Q&A", "🗨 Comments"])

# ===================================================================
# 1️⃣ Chat (with Persistent Memory + KakaoTalk Upload)
# ===================================================================
with tab1:
    st.header("Your Private GPT (with KakaoTalk Context)")

    username = st.text_input("Name (optional)", key="chat_user")
    question = st.text_input("Ask something")

    # ✅ KakaoTalk Upload
    uploaded_chat = st.file_uploader("📎 Upload KakaoTalk text file (.txt)", type=["txt"])
    kakao_memory_context = ""

    if uploaded_chat:
        content = uploaded_chat.read().decode("utf-8")
        parsed_msgs = KakaoTalkParser(content)

        if len(parsed_msgs) > 0:
            st.success(f"✅ Parsed {len(parsed_msgs)} messages successfully!")
            for msg in parsed_msgs[-5:]:
                st.markdown(f"🕓 *{msg[0]}* — **{msg[1]}**: {msg[2]}")

            kakao_memory_context = "\n".join(
                [f"{sender}: {text}" for (_, sender, text) in parsed_msgs]
            )
        else:
            st.warning("❌ Failed to parse KakaoTalk file. Please check encoding or format.")

    # --- Memory Persistent Logic ---
    if username:
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

        # ✅ DB에서 기존 memory 복원 (username 기준)
        cur.execute("SELECT memory_text FROM user_memory WHERE username = %s", (username,))
        row = cur.fetchone()
        if row and row[0]:
            try:
                messages_data = json.loads(row[0])
                st.session_state.memory.chat_memory.messages = []
                for msg in messages_data:
                    if msg["type"] == "human":
                        st.session_state.memory.chat_memory.add_user_message(msg["content"])
                    elif msg["type"] == "ai":
                        st.session_state.memory.chat_memory.add_ai_message(msg["content"])
            except Exception as e:
                print("⚠️ Memory restore failed:", e)
    else:
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

    # ✅ KakaoTalk 내용이 있다면 Memory에 반영
    if kakao_memory_context:
        if "kakao_loaded" not in st.session_state:
            st.session_state.memory.chat_memory.add_ai_message(
                f"Below is your KakaoTalk conversation history:\n{kakao_memory_context}\n"
                f"Now continue chatting naturally based on this context."
            )
            st.session_state.kakao_loaded = True

    # --- Define Chain with Memory ---
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template="""
        You are a friendly AI chatting partner. Use the previous KakaoTalk conversation (if available)
        and chat history to respond naturally and warmly.

        Conversation history:
        {chat_history}

        Human: {human_input}
        AI:
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.memory)

    if st.button("Send", key="chat_btn"):
        if not username:
            st.warning("Please enter your name first.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                response = chain.invoke({"human_input": question})
                answer = response["text"]

            st.markdown(f"**You:** {question}")
            st.markdown(f"**AI:** {answer}")

            # --- Save query log ---
            cur.execute("""
                INSERT INTO query_logs (username, mode, question, answer)
                VALUES (%s, %s, %s, %s)
            """, (username or "anonymous", "memory_chat", question, answer))
            conn.commit()

            # --- Memory -> JSON 변환 후 및 DB 백업 ---
            messages_to_save = []
            for m in st.session_state.memory.chat_memory.messages:
                if isinstance(m, HumanMessage):
                    messages_to_save.append({"type": "human", "content": m.content})
                elif isinstance(m, AIMessage):
                    messages_to_save.append({"type": "ai", "content": m.content})
            memory_json = json.dumps(messages_to_save, ensure_ascii=False)
            cur.execute("""
                INSERT INTO user_memory (username, memory_text, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (username)
                DO UPDATE SET memory_text = EXCLUDED.memory_text, updated_at = NOW();
            """, (username or "anonymous", memory_json))
            conn.commit()
            st.success("💾 Conversation saved to memory database!")


# ===================================================================
# 2️⃣ PDF Q&A
# ===================================================================
with tab2:
    st.header("PDF Question Answering")
    uploaded_file = st.file_uploader("Upload a PDF file")

    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_documents(pages)

        embeddings_model = OpenAIEmbeddings()
        db = Chroma.from_documents(texts, embeddings_model)

        cur.execute(
            "INSERT INTO embeddings_meta (username, filename, chroma_id) VALUES (%s, %s, %s)",
            ("default_user", uploaded_file.name, "local_chroma"),
        )
        conn.commit()

        st.success(f"Embedding completed: {uploaded_file.name}")
        st.write("---")

        pdf_q = st.text_input("Ask about this PDF")

        if st.button("Ask PDF", key="pdf_btn"):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            retriever = MultiQueryRetriever.from_llm(
                retriever=db.as_retriever(), llm=llm
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=db.as_retriever(),
            )

            with st.spinner("Analyzing PDF..."):
                result = qa_chain({"query": pdf_q})
                answer = result["result"]

            st.markdown(f"**Question:** {pdf_q}")
            st.markdown(f"**Answer:** {answer}")

            cur.execute("""
                INSERT INTO query_logs (username, mode, question, answer)
                VALUES (%s, %s, %s, %s)
            """, ("default_user", "pdf", pdf_q, answer))
            conn.commit()
            st.success("PDF Q&A saved to database!")


# ===================================================================
# 3️⃣ Comments
# ===================================================================
with tab3:
    st.header("Leave a Comment")

    username = st.text_input("Your name", key="comment_user")
    comment = st.text_area("Write your comment")

    if st.button("Submit", key="comment_submit"):
        if username and comment:
            cur.execute(
                "INSERT INTO comments (username, comment) VALUES (%s, %s)",
                (username, comment),
            )
            conn.commit()
            st.success("Comment saved!")

    cur.execute("SELECT username, comment, created_at FROM comments ORDER BY id DESC;")
    rows = cur.fetchall()
    for r in rows:
        st.markdown(f"**{r[0]}** · *{r[2]}*")
        st.write(r[1])
        st.divider()
