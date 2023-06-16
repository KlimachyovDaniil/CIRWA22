import telebot as tg
from telebot import types
from card_detection import get_card_img
from card_detector_canny import get_card_img_canny
from number_recognizer import recognize
import cv2
import imutils


bot = tg.TeleBot('6238993383:AAHZstnyHDZb_Jmx0Gf9FvDdEqoje-Oqlz4')


@bot.message_handler(content_types=['photo'])
def card_recognition(message):
    msg = bot.send_message(message.chat.id, 'Запрос обрабатывается...')
    photo_file = bot.get_file_url(message.photo[-1].file_id)
    file = imutils.url_to_image(photo_file)
    output = get_card_img(file)[0]
    output_img, card_number = recognize(output)
    if len(card_number) < 9:
        output = get_card_img_canny(file)
        output_img, card_number = recognize(output)
    with open('../bot/output.jpg', 'w'):
        cv2.imwrite('../bot/output.jpg', output_img)
    with open('../bot/output.jpg', 'rb') as photo:
        bot.delete_message(message.chat.id, msg.message_id)
        bot.send_photo(message.chat.id, photo, f'Номер карты: {card_number}')


@bot.message_handler(func=lambda m: True)
def echo_all(message):
    bot.reply_to(message, 'Отправьте фото карты с номером')


if __name__ == "__main__":
    bot.infinity_polling()
