# use pyTelegramBotAPI

from telebot import TeleBot

TOKEN = "6169099588:AAHFph2dMsDJnszbvUkjM8sRUghgLmR4Ui0"

bot = TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def on_start(message):
    bot.reply_to(message, "Hello, how are you doing?")
    bot.send_message(message.chat.id, "I'm a bot, please talk to me!")
    return

# send photo as message
image = open("./vu_dick.png", 'rb')
@bot.message_handler(commands=['photo'])
def msg3(message):
  bot.send_photo(message.chat.id, image)

bot.infinity_polling()