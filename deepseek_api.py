import requests

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
        """
        try:
            url = f"{self.api_base}/chat/completions"
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    api_key = "your_api_key_here"
    client = DeepSeekAPI(api_key)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "こんにちは！"}
    ]
    response = client.chat_completion(messages)
    if response:
        print(response)
