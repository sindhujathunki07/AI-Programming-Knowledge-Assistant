import os
import hashlib
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENTS_DIR = "documents"
DB_ROOT = "embedding_databases"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Programming Knowledge Assistant",
    page_icon="💻",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("💻 AI Programming Knowledge Assistant")

st.write(
    "RAG-powered programming assistant for learning programming concepts."
)


# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(DB_ROOT, exist_ok=True)


# ============================================================
# SESSION STATE
# ============================================================

if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db_path" not in st.session_state:
    st.session_state.db_path = None


# ============================================================
# PDF UPLOAD
# ============================================================

st.subheader("📄 Upload Programming PDF")

uploaded_files = st.file_uploader(
    "Upload your programming PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_files:

    # --------------------------------------------------------
    # Create hash for all uploaded files
    # --------------------------------------------------------

    hasher = hashlib.md5()

    for uploaded_file in uploaded_files:

        hasher.update(
            uploaded_file.name.encode()
        )

        hasher.update(
            uploaded_file.getvalue()
        )

    batch_hash = hasher.hexdigest()


    # --------------------------------------------------------
    # Process only new uploads
    # --------------------------------------------------------

    if st.session_state.processed_file_hash != batch_hash:

        with st.spinner("📚 Processing uploaded PDFs..."):

            try:

                all_documents = []


                # ------------------------------------------------
                # Create unique Chroma database directory
                # ------------------------------------------------

                db_path = os.path.join(
                    DB_ROOT,
                    batch_hash
                )

                os.makedirs(
                    db_path,
                    exist_ok=True
                )


                # ------------------------------------------------
                # Process every uploaded PDF
                # ------------------------------------------------

                for uploaded_file in uploaded_files:

                    # --------------------------------------------
                    # Save PDF
                    # --------------------------------------------

                    pdf_path = os.path.join(
                        DOCUMENTS_DIR,
                        uploaded_file.name
                    )

                    with open(
                        pdf_path,
                        "wb"
                    ) as f:

                        f.write(
                            uploaded_file.getvalue()
                        )


                    # --------------------------------------------
                    # Load PDF
                    # --------------------------------------------

                    loader = PyPDFLoader(
                        pdf_path
                    )

                    documents = loader.load()


                    # --------------------------------------------
                    # Check PDF
                    # --------------------------------------------

                    if not documents:

                        st.warning(
                            f"Could not read {uploaded_file.name}"
                        )

                        continue


                    # --------------------------------------------
                    # Split text
                    # --------------------------------------------

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=150
                    )

                    chunks = splitter.split_documents(
                        documents
                    )


                    # --------------------------------------------
                    # Add source filename
                    # --------------------------------------------

                    for chunk in chunks:

                        chunk.metadata["source"] = (
                            uploaded_file.name
                        )


                    # --------------------------------------------
                    # Add chunks to all documents
                    # --------------------------------------------

                    all_documents.extend(
                        chunks
                    )


                # ------------------------------------------------
                # Check if documents exist
                # ------------------------------------------------

                if not all_documents:

                    st.error(
                        "❌ No PDF content could be loaded."
                    )

                    st.stop()


                # ------------------------------------------------
                # Create embeddings
                # ------------------------------------------------

                embeddings = OllamaEmbeddings(
                    model=EMBEDDING_MODEL
                )


                # ------------------------------------------------
                # Create Chroma database
                # ------------------------------------------------

                vectorstore = Chroma.from_documents(
                    documents=all_documents,
                    embedding=embeddings,
                    persist_directory=db_path
                )


                # Release local reference
                # Do NOT delete the database
                del vectorstore


                # ------------------------------------------------
                # Save database information
                # ------------------------------------------------

                st.session_state.processed_file_hash = (
                    batch_hash
                )

                st.session_state.db_path = (
                    db_path
                )


                # ------------------------------------------------
                # Success message
                # ------------------------------------------------

                st.success(
                    f"✅ {len(uploaded_files)} PDF(s) "
                    "processed successfully!"
                )

                st.info(
                    "📚 You can now ask questions "
                    "about the uploaded PDFs."
                )


            except Exception as e:

                st.error(
                    "❌ Error processing PDFs:"
                )

                st.code(
                    str(e)
                )

                st.stop()


# ============================================================
# CHECK DATABASE
# ============================================================

if (
    st.session_state.db_path is None
    or not os.path.exists(
        st.session_state.db_path
    )
):

    st.info(
        "👆 Upload a programming PDF above to start."
    )

    st.stop()


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )


# ============================================================
# LLM
# ============================================================

@st.cache_resource
def get_llm():

    return ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )


# ============================================================
# LOAD CHROMA DATABASE
# ============================================================

@st.cache_resource
def get_database(db_path):

    embeddings = get_embeddings()

    database = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    return database


# ============================================================
# DATABASE
# ============================================================

try:

    db = get_database(
        st.session_state.db_path
    )

except Exception as e:

    st.error(
        "❌ Error loading Chroma database."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# RETRIEVER
# ============================================================

retriever = db.as_retriever(
    search_kwargs={
        "k": 5
    }
)


# ============================================================
# LOAD LLM
# ============================================================

try:

    llm = get_llm()

except Exception as e:

    st.error(
        "❌ Could not connect to Ollama."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
"""
You are an AI Programming Knowledge Assistant.

Answer the user's question using the uploaded PDF.

Rules:

1. Use the PDF information as the main source.
2. Explain concepts in simple beginner-friendly language.
3. Give examples when useful.
4. If the answer is not available in the PDF,
   say that it was not found in the uploaded document.
5. Do not invent information.
6. Use code blocks when showing programming code.
7. Keep the answer clear and well structured.

Context:

{context}

User Question:

{question}

Answer:
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")


    # --------------------------------------------------------
    # Knowledge Base
    # --------------------------------------------------------

    st.subheader("📚 Current Knowledge Base")

    if uploaded_files:

        for file in uploaded_files:

            st.success(
                f"📄 {file.name}"
            )

    else:

        st.info(
            "No PDF uploaded."
        )


    st.divider()


    # --------------------------------------------------------
    # AI Models
    # --------------------------------------------------------

    st.write("### 🤖 AI Models")

    st.write(
        f"LLM: `{LLM_MODEL}`"
    )

    st.write(
        f"Embedding: `{EMBEDDING_MODEL}`"
    )


    st.divider()


    # --------------------------------------------------------
    # Database status
    # --------------------------------------------------------

    st.write("### 🗄️ Database Status")

    if st.session_state.db_path:

        st.success(
            "Chroma database ready"
        )


# ============================================================
# CHAT SECTION
# ============================================================

st.divider()

st.subheader(
    "💬 Ask Your Programming Question"
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask a programming question..."
)


# ============================================================
# ANSWER QUESTION
# ============================================================

if user_question:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_question
        )


    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    # --------------------------------------------------------
    # Generate assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 Thinking..."
        ):

            try:

                # --------------------------------------------
                # Retrieve relevant documents
                # --------------------------------------------

                relevant_docs = retriever.invoke(
                    user_question
                )


                # --------------------------------------------
                # Create context
                # --------------------------------------------

                if relevant_docs:

                    context = "\n\n".join(
                        doc.page_content
                        for doc in relevant_docs
                    )

                else:

                    context = (
                        "No relevant information "
                        "was found in the uploaded PDF."
                    )


                # --------------------------------------------
                # Create prompt
                # --------------------------------------------

                formatted_prompt = prompt.invoke(
                    {
                        "context": context,
                        "question": user_question
                    }
                )


                # --------------------------------------------
                # Ask Llama
                # --------------------------------------------

                response = llm.invoke(
                    formatted_prompt
                )


                # --------------------------------------------
                # Get answer
                # --------------------------------------------

                answer = response.content


                # --------------------------------------------
                # Display answer
                # --------------------------------------------

                st.markdown(
                    answer
                )


                # --------------------------------------------
                # Save assistant message
                # --------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


            except Exception as e:

                st.error(
                    "❌ Error generating answer:"
                )

                st.code(
                    str(e)
                )