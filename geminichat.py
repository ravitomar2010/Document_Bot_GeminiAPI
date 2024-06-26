import streamlit as st
#from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#import google.generativeai as genai

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    #    chunks = data.load_and_split(text_splitter)
    return chunks


# create embeddings using GeminiEmbeddings() and save them in a Chroma vector store

def create_embeddings(chunks):
  if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")
  print("chnuks is ", chunks)
  # Initialize the embeddings model directly
  #GOOGLE_API_KEY = 'AIzaSyBEPGKZDgLcfbKBB3OxWirjPCRNBTQ4JmA'  # Replace with your actual API key
  #genai.configure(api_key=GOOGLE_API_KEY)
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
  # Pass the chunks list directly here
  #embedding = embeddings.embed_documents(chunks)  
  print("embedding is ", embeddings)
  vector_store = Chroma.from_documents(chunks, embeddings)
  return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    #from langchain.chains import RetrievalQA
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import PromptTemplate

    docs = vector_store.similarity_search(q)
    llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=1)
    
    template1 = """
    you are an AI Assistant of {context} , whenever any question {user_query} is asked you have to provided the more precise answer from {context} . 
    """

    template = ''' Answer the question as detailed as possible from the provided {context}, make sure to provide all the details, if the answer is not in
  provided context just say, "answer is not available in the context", don't provide the wrong answer
 
    ====================
    Context: {context}
    ====================
 
    Question: {user_query}
    '''

    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm
    #retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    #chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke({"context": docs, "sytem_prompt": prompt, "user_query":q })
    return answer['content']


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    import os

    # loading the Gemini api key from .env
    #from dotenv import load_dotenv, find_dotenv
    #load_dotenv(find_dotenv(), override=True)
    st.empty()
    st.image('OIP.jpeg')
    st.subheader('Question-Answering Application 🤖')
    with st.sidebar:
        # text_input for the Gemini API key (alternative to python-dotenv and .env)
        api_key = st.text_input('Gemini API Key:', type='password')
        if api_key:
            os.environ['GOOGLE_API_KEY'] = api_key
        GOOGLE_API_KEY=api_key
        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button
        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                print("data loaded ..")
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                print("creating embedding ....")

                vector_store = create_embeddings(chunks)
                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    # user's question text input widget
    q = st.text_input('Ask a question about the content of your file:')
    if q: # if the user entered a question and hit enter
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)

            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer)
            st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # the current question and answer
            value = f'Q: {q} \nA: {answer}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)

# run the app: streamlit run ./geminichat.py

