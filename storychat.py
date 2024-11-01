import os
import random

import pandas as pd
import streamlit as st

from openai import OpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from embeddings_aoss import AOSSEmbeddings
from streamlit_feedback import streamlit_feedback

from dotenv import load_dotenv
load_dotenv()

aoss_emb = AOSSEmbeddings()

def handle_feedback(fb_key, prompt, answer):
    st.toast("✔️ Feedback received!")
    file_path = "HumanFeedback.xlsx"
    df = pd.read_excel(file_path)
    row = [prompt, answer, st.session_state[fb_key]['score'], st.session_state[fb_key]['text']]
    new_row = pd.DataFrame([row], columns=df.columns)
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(file_path, index=False, engine="openpyxl")

def title_setup() -> None:
    """
    Set up the title, subtitle and information in the header
    """
    st.set_page_config(page_title="Stori Chat")
    st.markdown("""
    <style>
    .huge-font {
        font-size:40px !important;
    }
    .huge-font-italic {
        font-size:20px !important;
        font-style: italic;
    }
    .big-font {
        font-size:18px !important;
    }

    .big-font-italic {
        font-size:15px !important;
        font-style: italic;
        text-align: center;
    }

    .disclaimer-font {
        font-size:15px !important;
        font-style: italic;
        text-align: center;
    }
    .release-notes-font {
        font-size:15px !important;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    col1, mid, col2 = st.columns([4, 1, 20])
    with col1:
        st.image('media/stori_logo.png', width=100)
    with col2:
        st.markdown(
            f'<p><span class="huge-font"><b>Stori Chatbot</b></span> <span class="huge-font-italic">(Limited Support POC)</span></p>',
            unsafe_allow_html=True)
    subheader_text = "Hola! Bienvenido al Stori Chatbot (powered by <a href='https://www.linkedin.com/in/dr-alberto-beltran/'>Soullest</a>)"

    st.markdown(f'<p class="big-font">{subheader_text}</p>', unsafe_allow_html=True)


def sidepanel_setup() -> None:
    with st.sidebar:
        st.title('Ejemplos')
        st.info("""
        • ¿Cualés son las alianzas con comercios y proveedores de historiacard?\n
        • ¿Cuál es la visión de historiacard?\n
        • ¿Qué se puede hacer en la sección de Mi Perfil de la app de historiacard?\n
        • ¿Es posible acceder usando mi huella digital a mi app de historiacard?\n
        """)
        st.divider()
        st.title("Historial")
        if 'history' in st.session_state:
            with st.container():
                for msg in st.session_state.history.messages[1:]:
                    if msg.type == "human":
                        st.write(f"• {msg.content}", unsafe_allow_html=True)


def initial_setup() -> None:
    if 'setup_ready' not in st.session_state:
        st.session_state.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        st.session_state.model_id = os.environ["MODEL_ID"]
        st.session_state.history = StreamlitChatMessageHistory(key="chat_messages")
        st.session_state.history.add_user_message(os.environ["CONTEXT_PROMPT"])
        st.session_state.history.add_ai_message("Como te puedo ayudar?")
        st.session_state.messages =[
            {"role": "system", "content": os.environ["CONTEXT_PROMPT"]},
            {"role": "assistant", "content": "Como te puedo ayudar?"}
        ]
        ev_cols = ['Prompt', 'Response', 'Expected Answer', 'Source_1', 'Source_2', 'Source_3', 'Human Feedback']
        st.session_state.evaluation_df = pd.DataFrame(columns=ev_cols)
        st.session_state.setup_ready = 0


def main() -> None:
    """
    This function will be reach as the start point to the chat
    """
    title_setup()

    if "widget_key" not in st.session_state:
        st.session_state["widget_key"] = str(random.randint(1, 1000000))

    sidepanel_setup()

    initial_setup()
    last_prompt = ''
    for idx, msg in enumerate(st.session_state.history.messages[1:], start=1):
        if msg.type == 'ai':
            st.chat_message(msg.type, avatar='media/stori_logo.png').write(msg.content, unsafe_allow_html=True)
            if idx > 1:
                with st.form(f'form{str(random.randint(1, 1000000))}'):
                    form_key = f'fb_k{str(random.randint(1, 1000000))}'
                    streamlit_feedback(feedback_type="thumbs",
                                       optional_text_label="[Optional] Please provide an explanation",
                                       align="flex-start",
                                       key=form_key)
                    st.form_submit_button('Save feedback', on_click=handle_feedback, args=(form_key, last_prompt, msg.content))
        else:
            st.chat_message(msg.type).write(msg.content, unsafe_allow_html=True)
            last_prompt = msg.content


    # Get the prompt from the user
    if prompt := st.chat_input():

        st.chat_message("user").write(prompt)
        with st.spinner('Un momento...'):
            config = {"configurable": {"session_id": "any"}}
            emb_question = f"""
            Summarize the following question in only the 3 to 5 most relevant words. 
            Answer using only 3 to 5 words, do not add anything else.
            Question: {prompt}
            """
            completion = st.session_state.client.chat.completions.create(
                model=st.session_state.model_id,
                messages=[
                    {"role": "system",
                     "content": "Eres un asistente que resume informacion"},
                    {
                        "role": "user",
                        "content": emb_question
                    }
                ]
            )

            emb_abs = completion.choices[0].message.content
            print(f"{'-'*20}\nEmb abstract: {emb_abs}\n{'-'*20}")
            emb_list = aoss_emb.query(question=emb_abs, k=3)
            emb_info = '\n'.join(emb_list)
            full_question = f"""Responde la siguiente pregunta:
            {prompt}
            
            Puedes usar la siguiente información como guia.
            {emb_info}
            """
            print(f"{'*'*20}\n{full_question}\n{'*'*20}")
            placeholder = st.empty()
            full_response = ''

        st.session_state.messages.append({"role": "user", "content": full_question})
        st.session_state.history.add_user_message(prompt)
        for delta in st.session_state.client.chat.completions.create(
                model=st.session_state.model_id,
                temperature=0,
                top_p=0.2,
                messages=st.session_state.messages,
                stream=True):
            full_response += str(delta.choices[0].delta.content)
            placeholder.chat_message("ai", avatar='media/stori_logo.png').write(full_response)
        # message = {"role": "assistant", "content": response}
        # st.session_state.visible_messages.append(message)
        # for chunk in st.session_state.chain_with_history.stream({"question": prompt}, config):
        #     full_response += chunk
        #     placeholder.chat_message("ai", avatar='media/stori_logo.png').write(full_response)


        placeholder.chat_message("ai", avatar='media/stori_logo.png').write(full_response, unsafe_allow_html=True)

        placeholder.chat_message("ai", avatar='media/stori_logo.png').write(full_response, unsafe_allow_html=True)


        st.session_state.history.add_ai_message(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_question})

        row = [prompt, full_response, '', emb_list[0], emb_list[1], emb_list[2]]
        file_path = "metrics.xlsx"
        df = pd.read_excel(file_path)
        new_row = pd.DataFrame([row], columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(file_path, index=False, engine="openpyxl")
        st.rerun()

if __name__ == "__main__":
    main()

