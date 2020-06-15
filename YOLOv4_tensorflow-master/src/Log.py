# coding:utf-8
# log
import config
from os import path
import os
from utils import tools

# add a log message
def add_log(content):
    if not path.isdir(config.log_dir):
        os.mkdir(config.log_dir)
        add_log("message:create folder '{}'".format(config.log_dir))
    log_file = path.join(config.log_dir, config.log_name)
    print(content)
    tools.write_file(log_file, content, True)
    return

# add a loss value
def add_loss(value):
    if not path.isdir(config.log_dir):
        os.mkdir(config.log_dir)
        add_log("create folder '{}'".format(config.log_dir))
    loss_file = path.join(config.log_dir, config.loss_name)
    tools.write_file(loss_file, value, False)
    return