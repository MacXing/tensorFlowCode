# -*- coding: utf-8 -*- 
# @Time : 2018/8/2 16:03 
# @Author : Allen 
# @Site :
import os


def get_dir():
    return os.path.abspath(os.getcwd())+os.sep

def is_dir(path):
    if os.path.exists(path):
        os.mkdir(path)
        return path
    else:
        return path