from deepseek_api import DeepSeekAPI

# APIキーを設定
api_key = "your_api_key_here"  # 実際のAPIキーに置き換えてください

# クライアントの初期化
client = DeepSeekAPI(api_key)

# メッセージの作成
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "こんにちは！"}
]

# API呼び出し
response = client.chat_completion(messages)
if response:
    print(response)import requests

class DeepSeekAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_base = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat_completion(self, messages, model="deepseek-chat", temperature=0.7):
        """
        DeepSeek APIを使用してチャット応答を生成します。
        
        Args:
            messages (list): 会話メッセージのリスト
            model (str): 使用するモデル名
            temperature (float): 生成の多様性を制御するパラメータ（0.0〜1.0）
        
        Returns:
            dict: APIレスポンス
        """
        endpoint = f"{self.api_base}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling DeepSeek API: {e}")
            return None

# 使用例
if __name__ == "__main__":
    # APIキーを設定（実際の使用時は環境変数から取得することを推奨）
    api_key = "your_api_key_here"
    
    # DeepSeek APIクライアントのインスタンスを作成
    client = DeepSeekAPI(api_key)
    
    # テストメッセージ
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you?"}
    ]
    
    # API呼び出しのテスト
    response = client.chat_completion(messages)
    if response:
        print("API Response:", response)
