from flask import Flask, request, abort
import os
import json
import datetime
from keras.backend import clear_session

# from __future__ import print_function
# from keras.callbacks import LambdaCallback
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras.layers import LSTM
# from keras.optimizers import RMSprop
# from keras.utils.data_utils import get_file
# import numpy as np

from dialog_generate import generate

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

app = Flask(__name__)

# 環境変数取得
# YOUR_CHANNEL_ACCESS_TOKEN = os.environ["YOUR_CHANNEL_ACCESS_TOKEN"]
# YOUR_CHANNEL_SECRET = os.environ["YOUR_CHANNEL_SECRET"]

YOUR_CHANNEL_ACCESS_TOKEN = "Xvfv904iY8pzJJEEfaXhwz+IkTN4L9Bct/RgYiL3MZJ7nvC40PcmjDfOYHsvmxVUzKpJ97H/K38N2Ncd5DXPZYfFFeuOBy7+DQNnFJkyV744BPosXKs2zjGJgY2nXPjEoM1VlfODRDlvhpo0L8IIngdB04t89/1O/w1cDnyilFU="
YOUR_CHANNEL_SECRET = "f2aa46d3314b962079f34b9199f3fe49"

line_bot_api = LineBotApi(YOUR_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(YOUR_CHANNEL_SECRET)

result = ""
# global result


@app.route("/callback", methods=["POST"])
def callback():
    # clear_session()
    # get X-Line-Signature header value
    signature = request.headers["X-Line-Signature"]

    # get request body as text
    body = request.get_data(as_text=True)
    # print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Request body: " + body)
    # app.logger.info("Request body: " + body)

    # dialog_generate.pyのXXX関数を実行
    global result
    result = generate()
    print("result" + result)
    print("body: " + body)

    # message_event = json.loads(event["body"])["events"][0]
    # reply_token = message_event["replyToken"]
    # # message_text = message_event['message']['text']
    # message_text = result

    # requests.post(
    #     "https://api.line.me/v2/bot/message/reply",
    #     data=json.dumps(
    #         {
    #             "replyToken": reply_token,
    #             "messages": [{"type": "text", "text": message_text.upper()}],
    #         }
    #     ),
    #     headers={
    #         # TODO: Put your channel access token in the Authorization header
    #         "Authorization": "Bearer YOUR_CHANNEL_ACCESS_TOKEN_HERE",
    #         "Content-Type": "application/json",
    #     },
    # )

    # body["events"]["message"]["text"] = result
    # body.events.message.text = result
    # print('body["events"]["message"]["text"]:' + body["events"]["message"]["text"])
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    print("global result: " + result)
    print("TextMessage: " + str(TextMessage))
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result))


if __name__ == "__main__":
    #    app.run()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
