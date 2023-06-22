import streamlit as st
from streamlit_chat import message
import pinecone
import openai
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import os
import re 
openai.api_key = st.secrets["openai_key"]


st.set_page_config(page_title="MindyAI - An LLM-powered M1 Chatbot")

hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)















with st.sidebar:
    st.title('🤗 M1 Mindy Demostration')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - Streamlit
    - OpenAI
    - PineCone 
    ''')
    add_vertical_space(5)



# Connect to the index and view index stats
col1,col2,col3 = st.columns(3)

with col2:
    st.image('mindycat.png',width = 200)


user_message  = st.text_input("Ask Mindy Anything: ", "", key="input")

embed_model = "text-embedding-ada-002"

combi_query = ''
# system message to assign role the model
system_msg = f"""You are a helpful and excited customer service bot for M1, a telecommunication company based in singapore. 
Answer questions in full based on the context provided if you do not know, say you dont know and try to give something similar to what is requested. If you are asked for any comparison or differences or similarities, present your output in
a tabular format. If you are giving any prices, please double check from sources to make sure that the price is correct and accurate. 
"""
def display_with_sources(response_content, matches):
    # Only keep alphanumeric characters, spaces, or punctuation
    #response_content = ''.join(ch for ch in response_content if ch.isalnum() or ch.isspace() or ch in string.punctuation)
    response_content = response_content.replace("$", "SGD ")
    response = f"{response_content}\n\nSources:\n"
    count = 0  # Counter to track the number of URLs added

    for match in matches:
        if count < 3:  # Limit the number of URLs to 3
            title = match['metadata']['title'].title()
            url = match['metadata']['url']
            response += f"- [{title}]({url})\n"
            count += 1
        else:
            break  # Break the loop if 3 URLs have been added
    colored_header(label='', description='', color_name='orange-30')
    print(response)
    st.write(response)



if st.button('Ask Mindy!'):
    with st.spinner('Mindy is thinking...'):

        index_name = 'm1full'

        # Initialize connection to Pinecone
        pinecone.init(api_key=st.secrets["pinecone_key"], environment = 'us-west1-gcp')

        index = pinecone.Index(index_name)
        embed_query = openai.Embedding.create(
            input=[user_message],
            engine=embed_model
        )

        # retrieve from Pinecone
        query_embeds = embed_query['data'][0]['embedding']

        # get relevant contexts (including the questions)
        response = index.query(query_embeds, top_k=5, include_metadata=True)
        matches = response['matches']

        # get list of retrieved text
        contexts = [item['metadata']['text'] for item in response['matches']]

        # concatenate contexts and user message to generate augmented query
        augmented_query = " --- ".join(contexts) + " --- " + user_message
        #combi_query = '' + augmented_query
        #combi_query =  ' '.join(combi_query.split()[:300])

        chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": augmented_query}
        ],
        temperature  = 0.2
        )

        assistant_message = chat['choices'][0]['message']['content']
        try:
            messages.append({"role": "assistant", "content": assistant_message})
        except:
            print('None')
        # print(assistant_message)
        display_with_sources(assistant_message, matches)
