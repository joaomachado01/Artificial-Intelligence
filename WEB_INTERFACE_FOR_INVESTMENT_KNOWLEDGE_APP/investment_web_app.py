###############################################
###############   Web APP   ###################
###############################################

# Imports
import os
import re
import streamlit as st
import requests
import json
import warnings
warnings.filterwarnings("ignore")

### Setting front end page ###

# Setting page title and configs
st.set_page_config(page_title="Investment project", page_icon=":chart_with_upwards_trend:", layout="centered")
st.image("assets/banner.png", width="stretch")

# Setting title for the Streamlit App
st.title('_Investment Knowledge App - Stocks_')
st.title('_:blue[Knowledge Retrieval with Generative AI and RAG]_')

# Creates a text input box
question = st.text_input("Ask a question for the AI to give you the right context content:", "")

# Checks if Send button was applied
if st.button("Send"):

    # For when no question is in the text box
    if not question.strip():
        st.warning("Please type a question.")
        st.stop()
    
    # Show the question
    st.write("The question was: \"", question+"\"")

    # Define the API URL
    api_url = os.getenv("API_URL", "http://127.0.0.1:8000/invest_api")

    # Creates the payload
    payload = json.dumps({"query": question})

    # Creates headers
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    # Post request to the API with error handling
    try:
        response = requests.request("POST", api_url, headers=headers, data=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        st.stop()

    # Get response content
    response_content = json.loads(response.text)['answer']

    # Creates a regex to find expressions that has the references numbers (of the docs)
    rege = re.compile(r"\[Document\ [0-9]+\]|\[[0-9]+\]")

    # Gets the references expressions
    expressions = rege.findall(response_content)

    # Initiate list to hold refs retrieved by the expressions
    number_refs_list = []

    # Extract number refs from expressions
    for exp in expressions:
        number_refs_list = number_refs_list + [int(exp_token) for exp_token in re.findall(r'\b\d+\b', exp)]

    # Dedupe entries
    number_refs_list = list(dict.fromkeys(number_refs_list))

    # Show response using markdow format
    st.markdown(response_content)

    # Gets documents from the response context key
    documents = json.loads(response.text)["context"]

    # Documents list related to the refs
    refs_docs_list = []

    # Add the docs related to the refs
    for ref in number_refs_list:
        for doc in documents:
            if int(doc['id']) == ref:
                refs_docs_list.append(doc)

    # Ids for the download buttons (1 button for doc ref)
    doc_id = 12991345567899999

    # Show doc content and download for each document
    for doc in refs_docs_list:

        # Creates expander for each document
        with st.expander(str(doc['id'])+" - "+doc['path']):

            # Show document content
            st.write(doc['content'])

            # Opens document and create download button
            try:
                with open(doc['path'], 'rb') as f:
                    
                    st.download_button("Download file", f, file_name = doc['path'].split('/')[-1], key = doc_id)
            
            except FileNotFoundError:
                st.caption("Original file not found")
                    
            # Increments the doc ids (when the refs is more then 1)
            doc_id = doc_id +1 
            

