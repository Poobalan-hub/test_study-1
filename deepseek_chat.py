import streamlit as st
from deepseek_api import DeepSeekAPI
import os
import signal

# ページ設定
st.set_page_config(
    page_title="AI Medi Chat",
    page_icon="🏥",
    layout="wide"
)

# サイドバー
with st.sidebar:
    st.title("設定")
    
    # モデル選択
    model = st.selectbox(
        "モデルを選択",
        ["deepseek-chat", "deepseek-coder"]
    )
    
    # 下部にスペースを追加してボタンを下に配置
    st.markdown("<br>" * 10, unsafe_allow_html=True)
    
    # アプリ終了ボタン
    if st.button("アプリ終了", type="primary"):
        os.kill(os.getpid(), signal.SIGTERM)

# APIキーの設定
API_KEY = "sk-fb5ed929cd134354b20d4557c194e651"

# DeepSeek APIクライアントの初期化
client = DeepSeekAPI(API_KEY)

# メインエリア
# タイトル
st.markdown("""
    <h1 style='text-align: center; color: #2e4053; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>
        AI Medi Chat
    </h1>
    """, unsafe_allow_html=True)
st.markdown("---")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful medical assistant."}
    ]

# チャット履歴の表示
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# ユーザー入力
if prompt := st.chat_input("メッセージを入力してください"):
    # ユーザーメッセージの表示
    with st.chat_message("user"):
        st.write(prompt)
    
    # メッセージの追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # AIの応答を生成
    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            response = client.chat_completion(
                st.session_state.messages,
                model=model,
                temperature=0.7  # デフォルト値を設定
            )
            if response and "choices" in response:
                ai_message = response["choices"][0]["message"]["content"]
                st.write(ai_message)
                st.session_state.messages.append({"role": "assistant", "content": ai_message})
