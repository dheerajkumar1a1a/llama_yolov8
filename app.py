from tempfile import NamedTemporaryFile
import os
import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool
import stat
import os

directory_path = 'C:\\Users\\Dheeraj\\Downloads\\aree-main\\'

# Check current directory permissions
current_permissions = os.stat(directory_path)
print(f"Current Permissions: {current_permissions.st_mode}")

# Add write permissions for the user
os.chmod(directory_path, current_permissions.st_mode | stat.S_IWUSR)
# Replicate Credentials
# with st.sidebar:
#     st.title('🦙💬 Llama 2 Chatbot')
#     st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')
#     if 'REPLICATE_API_TOKEN' in st.secrets:
#         st.success('API key already provided!', icon='✅')
#         replicate_api = st.secrets['REPLICATE_API_TOKEN']
#     else:
#         replicate_api = st.text_input('Enter Replicate API token:', type='password')
#         if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
#             st.warning('Please enter your credentials!', icon='⚠️')
#         else:
#             st.success('Proceed to entering your prompt message!', icon='👉')
#     os.environ['REPLICATE_API_TOKEN'] = replicate_api

##############################
### initialize agent #########
##############################
tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key="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",
    temperature=0,
    model_name="gpt-3.5-turbo"
)

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
        print(f"Temporary Image Path: {image_path}")
        # with open(image_path, 'rb') as binary_file:
        #     file_content = binary_file.read()
        #     print(f"File Content (Binary): {file_content}")

        # write agent response
        if user_question and user_question != "":
            with st.spinner(text="In progress..."):
                response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))
                st.write(response)
