from tempfile import NamedTemporaryFile
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import streamlit as st
import replicate  # Import LLama2's replicate function
from langchain.llms import Replicate
from tools import ImageCaptionTool, ObjectDetectionTool
import os 

##############################
### initialize agent #########
##############################

tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    if 0: #'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = 'r8_1ZviNlfE94ZzfocsHPr9tdZoCEXMJu31IDM3R'

llm = Replicate(
    model='a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea',
    model_kwargs={"temperature": 0.01, "max_length": 500, "top_p": 1},
)




# st.subheader('Models and parameters')
# selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
# if selected_model == 'Llama2-7B':
#     llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
# elif selected_model == 'Llama2-13B':
#     llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
# temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
# top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
# max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
# st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)

# set title
st.title('Ask a question to an image')

# set header
st.header("Please upload an image")

# upload file
file = st.file_uploader("Upload", type=["jpeg", "jpg", "png"])

if file:
    # display image
    st.image(file, use_column_width=True)

    # text input
    user_question = st.text_input('Ask a question about your image:')

    ##############################
    ### compute agent response ###
    ##############################
    with NamedTemporaryFile(dir='.') as f:
        f.write(file.getbuffer())
        image_path = f.name

        # write agent response
        if user_question and user_question != "":
            with st.spinner(text="In progress..."):
                response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))
                st.write(response)
