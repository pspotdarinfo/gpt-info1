import streamlit as st  # pip install streamlit
import streamlit_authenticator as stauth  # pip install streamlit-authenticator==0.1.5
from streamlit_extras.add_vertical_space import add_vertical_space
import os
import string
import random
import pandas as pd
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.vectorstores import Chroma
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time
os.chdir(r"C:\Users\pp_info1\Downloads\streamlit_llm")
import database as db
from data_preparation import *

from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

def save_uploadedfile(uploadedfile):
     with open(os.path.join("uploadedfiles",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
         print("Uploaded Files:{}".format(uploadedfile.name))
     return st.success("Uploaded Files:{}".format(uploadedfile.name))

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="InfoCepts LLM", page_icon=":scroll:", layout="wide")

# --- USER AUTHENTICATION ---
import psycopg2
#establishing the connection
conn = psycopg2.connect(
   database="postgres", user='postgres', password='postgres', host='127.0.0.1', port= '5432'
)
#Setting auto commit false
conn.autocommit = True
#Creating a cursor object using the cursor() method
cursor = conn.cursor()
#Retrieving data
cursor.execute('''SELECT * from llm_login''')
#Fetching 1st row from the table
users = cursor.fetchall();
#Commit your changes in the database
conn.commit()
#Closing the connection
conn.close()

usernames = [user[0] for user in users]
names = [user[1] for user in users]
hashed_passwords = [user[2] for user in users]
authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "InfoCepts LLM", "llm", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status == True:
    # ---- SIDEBAR ----
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")
    #datetime_str = str(datetime.now())
    #insert_user_activity(username,name,datetime_str,"gpt4all_nzoozy")
    
    #add_vertical_space(5)
    import streamlit as st
    from streamlit_option_menu import option_menu

    with st.sidebar:
        #selected = option_menu("Menu", ["Inputs", 'Model Parameters', 'Ask your LLM'], 
        #    icons=['house', 'gear', 'question'], default_index=0)
        
        selected = option_menu(None, ["Home", "Inputs", "Model Parameters", 'Ask your LLM', 'Show my Data'], 
        icons=['house', 'plus-circle','gear' ,"pencil-square", "database"], 
        menu_icon="cast", default_index=0, orientation="Vertical")

    if selected == "Home":
        st.title("About")
        st.write("Placeholder")

    elif selected == "Inputs":
        
        #st.session_state['df_inputs'] = pd.DataFrame(columns=["option_model","option_embedd","path_input"])
        
        if "df_inputs" not in st.session_state:
            st.session_state['df_inputs'] = pd.DataFrame(columns=["option_model","option_embedd","path_input"])
        st.title("Inputs")
        file = st.file_uploader("Upload your Files")
        if file is not None:
            st.success("Uploaded Files:{}".format(file.name))
        with st.form("my_form"):
            #st.title("Inputs")
            models = [*list(os.listdir(r"C:\Users\pp_info1\Downloads\streamlit_llm\Models"))]
            embedd=   ['sentence-transformers/all-MiniLM-L6-v2']
            option_model = st.selectbox('Choose your model',models)
            
            #st.write('You selected:', option_model)
            
            option_embedd = st.selectbox('Choose your embeddings transformer',embedd)
            
            #st.write('You selected:', option_embedd)
            
            path_input = r'C:\Users\pp_info1\Downloads\streamlit_llm\uploadedfiles'
            #path_input = st.text_input('Please enter the file path',r'C:\Users\pp_info1\Downloads\streamlit_llm\uploadedfiles')
            #uploaded_files = st.file_uploader("Choose a PDF(s) to feed the model", accept_multiple_files=True)
            def onTrainModel():
                
                st.session_state['df_inputs'] = pd.DataFrame(columns=["option_model","option_embedd","path_input"])
            
                data = {
                'option_model':[option_model],
                'option_embedd': [option_embedd],
                'path_input':[path_input],
                }
                
                if path_input == '' or option_model == '' or option_embedd == '':
                    st.write("All Inputs should be provided..")
                else:
                    st.session_state['df_inputs'] = pd.concat([st.session_state['df_inputs'],pd.DataFrame(data)])
                    st.session_state['df_inputs'] = st.session_state['df_inputs'].drop_duplicates()
                    st.session_state['files'] = file
                
                progress_text = "Model Training in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                if file is not None:
                    print("file uploading")
                    save_uploadedfile(file)
                    print("file uploaded")
                # Embeddings
                embeddings = HuggingFaceEmbeddings(model_name=option_embedd)
                for percent_complete in range(50):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                # Create and store locally vectorstore
                print("Creating new vectorstore...")
                texts = process_documents(source_directory = path_input, chunk_size = 1024, chunk_overlap = 64)
                print("Creating embeddings. May take some minutes...")
                store = Chroma.from_documents(texts, embeddings)
                print("Embeddings created successfully")
                st.session_state['store'] = store
                #progress_text = "Model Training in progress. Please wait."
                #my_bar = st.progress(0, text=progress_text)
                df_input = st.session_state['df_inputs']
                option_model1 = df_input['option_model'][0]
                option_model1 = os.path.join(os.getcwd(), "Models",option_model1)
                # Callbacks support token-wise streaming
                callbacks = [StreamingStdOutCallbackHandler()]
                
                print("Loading Model")
                #llm = GPT4All(model=model, backend="gptj", callbacks=callbacks, verbose=False, n_ctx=2048)
                llm = GPT4All(model=option_model1, n_ctx=2048, backend='gptj', n_batch=8, callbacks=callbacks, verbose=True, temp = 0.7, n_predict = 4096, top_p = 0.1, top_k = 40, repeat_penalty = 1.18)

                
                print("Model Loaded")
                st.session_state['llm'] = llm
                for percent_complete in range(50,100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                st.write("Training Complete!")
            
            st.form_submit_button("Finetune Model", on_click = onTrainModel)   

        
    elif selected == "Model Parameters":
        st.title("Model Parameters")
        st.write("Placeholder")
        #if "df_inputs" not in st.session_state:
        #    st.session_state['df_inputs'] = pd.DataFrame(columns=["option_model","option_embedd","path_input"])

        
    elif selected == "Ask your LLM":
        st.title("Ask you LLM")
        #st.write(st.session_state['df_inputs'].drop_duplicates())
        
        
        method = st.radio("Select a Method", ('RAG', 'RetrievalQA'), horizontal = True)
        question_input = st.text_input('Please enter your question here')
        

        def getresponse_rqa(model, store, question, method, llm):
            
            qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=store.as_retriever(search_kwargs={"k": 2}),
                    return_source_documents=True,
                    verbose=False,
                )
            
            
            start = time.time()
            response = qa(question)
            answer, docs = response['result'], response['source_documents']
            end = time.time()
    
            # Print the result
            st.write("Question:")
            st.write(response['query'])
            st.write("Answer:")
            st.write(answer)
            st.write("Method:")
            st.write(method)
            st.write(f"\n> Answer (took {round(end - start, 2)} s.):")
            # Print the relevant sources used for the answer
            for document in docs:
                st.write("\n> " + document.metadata["source"] + ":")
                st.write(document.page_content)

            datetime_str = str(datetime.now())
            count = ''.join(random.choices(string.ascii_uppercase +
                                          string.digits, k=7))
            db.insert_user_activity(str(count),username,name,datetime_str,model, method,response['query'], response['result'])

        
        def getresponse_rag(model, store, question, method, llm):
            docs = store.similarity_search(question,k=2)
                        
            #prompt_template = """Answer based on context:\n\n{context}\n\n{question}"""
            #PROMPT = PromptTemplate(template=prompt_template, input_variables=[ "context","question"]) 
            #chain = load_qa_chain(llm=llm, prompt=PROMPT)
            #result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)["output_text"]
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            
            start = time.time()
            result = chain.run(input_documents=docs, question=question)
            end = time.time()
            
            
            datetime_str = str(datetime.now())
            count = ''.join(random.choices(string.ascii_uppercase +
                                         string.digits, k=7))
            db.insert_user_activity(str(count),username,name,datetime_str,model, method,question, result)

            
            st.write("Question:")
            st.write(question)
            st.write("Answer:")
            st.write(result)
            st.write("Method:")
            st.write(method)
            st.write(f"\n> Answer (took {round(end - start, 2)} s.):")

        if question_input:
            df_input = st.session_state['df_inputs']
            option_model = df_input['option_model'][0]
            option_model = os.path.join(os.getcwd(), "Models",option_model)
            #option_embedd = df_input['option_embedd'][0]
            #path_input = df_input['path_input'][0]
            store = st.session_state['store']
            llm = st.session_state['llm']
            if question_input=='':# or path_input=='':
                st.write('The question prompt/path is empty')
            else:
                print("Generating Answer....")
                
                if method == 'RetrievalQA':
                    getresponse_rqa(model = option_model,store=store, question = question_input, method=method, llm=llm)  
                elif method == 'RAG':
                    getresponse_rag(model = option_model,store=store, question = question_input, method=method, llm=llm)
                print("\nAnswer provided..")
    
    elif selected == 'Show my Data':
         st.title('My Data')
         data = db.fetch_user_activity()
         ans = pd.DataFrame.from_dict(data)
         ans.columns = ['key','username','name','datetime','model','method','prompt','result']
         ans = ans[ans['name']==name]
         ans = ans[['key','username','name','datetime','model','method','prompt','result']]
         st.write(ans[['key','username','name','datetime','model','method','prompt','result']])
        
        

