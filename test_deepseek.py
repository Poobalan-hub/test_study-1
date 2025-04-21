import sys
import json
from deepseek_api import DeepSeekAPI

# 文字エンコーディングをUTF-8に設定
sys.stdout.reconfigure(encoding='utf-8')

# APIキーを設定
api_key = "sk-fb5ed929cd134354b20d4557c194e651"

# DeepSeek APIクライアントのインスタンスを作成
client = DeepSeekAPI(api_key)

# テストメッセージ
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "こんにちは！今日の天気を教えてください。"}
]

# API呼び出しのテスト
response = client.chat_completion(messages)
if response:
    print("API Response:", json.dumps(response, ensure_ascii=False, indent=2))
