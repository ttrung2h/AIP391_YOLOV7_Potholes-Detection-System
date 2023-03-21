import requests
class ChatBot:
    def __init__(self,token,chat_id) -> None:
        self.token = token
        self.chat_id = chat_id
    def sendPhoto(self,cap,file):
        r = requests.post("https://api.telegram.org/bot{}/sendPhoto?chat_id={}&caption={}".format(self.token, self.chat_id, cap), files=file)
        print(r.status_code, r.reason)

    def sendText(self,text):
        r = requests.post("https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}".format(self.token, self.chat_id, text))
        print(r.status_code, r.reason)
    
