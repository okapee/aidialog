from flask import Flask, request, abort
import os
from datetime import datetime

# この３つで文字別／単語別／単語別（word2vec)
# from dialog_generate_by_char import generate
from dialog_generate_by_word_onehot import generate

# from dialog_generate_by_word2vec import generate


from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

app = Flask(__name__)

# 環境変数取得
YOUR_CHANNEL_ACCESS_TOKEN = os.environ["YOUR_CHANNEL_ACCESS_TOKEN"]
YOUR_CHANNEL_SECRET = os.environ["YOUR_CHANNEL_SECRET"]

line_bot_api = LineBotApi(YOUR_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(YOUR_CHANNEL_SECRET)

result = ""


@app.route("/callback", methods=["POST"])
def callback():
    global result
    # get X-Line-Signature header value
    signature = request.headers["X-Line-Signature"]

    # get request body as text
    body = request.get_data(as_text=True)
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Request body: " + str(body))
    print(
        datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        "Request body type: " + str(type(body)),
    )
    app.logger.info("Request body: " + body)

    if "文章生成モード" in body:
        print("文章生成モード")

        result = generate()
    elif "対話モード" in body:
        print("対話モード")

        result = "対話モードはまだないよ…　もう少し待ってね！"

    # dialog_generate.pyのgenerate関数を実行

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
