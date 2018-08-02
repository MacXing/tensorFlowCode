# -*- coding: utf-8 -*- 
# @Time : 2018/8/2 8:46 
# @Author : Allen 
# @Site :  生成验证码

from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


class Captcha():
    def __init__(self, captcha_size=4):
        self.number = [str(i) for i in range(10)]
        self.alphabet = [chr(i) for i in range(97, 123)]
        self.ALPHABET = [chr(i) for i in range(65, 91)]
        self.char_set = self.number + self.alphabet + self.ALPHABET
        self.captcha_size = captcha_size

    def random_captcha_text(self):
        return ''.join([random.choice(self.char_set) for i in range(self.captcha_size)])

    def get_captcha_text_and_image(self):
        image = ImageCaptcha()

        captcha_text = self.random_captcha_text()
        captcha = image.generate(captcha_text)
        captcha_image = Image.open(captcha)
        captcha_image = np.array(captcha_image)
        return captcha_text, captcha_image

    def get_len_char_set(self):
        return len(self.char_set)


if __name__ == '__main__':
    captcha = Captcha()
    text, image = captcha.get_captcha_text_and_image()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
    plt.show()
