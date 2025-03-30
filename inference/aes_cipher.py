#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import time
import numpy as np
from test_vectors import text2vector, get_vectors
from basic_block_parallel import key_expansion
from datetime import datetime

model_path = "YOUR_PATH"
model_path = "models/aes/models/ecb_chipher"

def inference():
    plaintext = '6bc1bee22e409f96e93d7e117393172a'
    plaintext = text2vector(plaintext)
    block_num = 10
    plaintext = plaintext * block_num
    plaintext = tf.constant(plaintext, dtype=tf.int32)
    plaintext = tf.reshape(plaintext, shape=(block_num, 16))

    length = tf.constant(plaintext.shape[0], dtype=tf.int32)

    key = '2b7e151628aed2a6abf7158809cf4f3c'
    key = text2vector(key)
    round_keys = key_expansion(key)
    round_keys = tf.constant(round_keys, dtype=tf.int32)

    # load model
    start_time = time.time()
    cipher = tf.saved_model.load(model_path)
    model_load_time = time.time() - start_time
    res = cipher(plaintext, round_keys)
    inference_time = time.time() - start_time - model_load_time

    # print(cipher())
    print("tflite model outputs: ", res)
    print("\t model load time used:", model_load_time)
    print('\t inference time used:', inference_time)

def validate_correctness():
    key, plaintext, ciphertext = get_vectors('ecb_128', single_block=False)

    # plaintext = '6bc1bee22e409f96e93d7e117393172a'
    plaintext = text2vector(plaintext)
    block_num = 1
    plaintext = plaintext * block_num
    plaintext = tf.constant(plaintext, dtype=tf.int32)
    plaintext = tf.reshape(plaintext, shape=(block_num * 4, 16))

    length = tf.constant(block_num * 4, dtype=tf.int32)

    # process ciphertext
    ciphertext = text2vector(ciphertext)
    ciphertext = tf.constant(ciphertext, dtype=tf.int32)
    ciphertext = tf.reshape(ciphertext, shape=(block_num * 4, 16))

    key = '2b7e151628aed2a6abf7158809cf4f3c'
    key = text2vector(key)
    round_keys = key_expansion(key)
    round_keys = tf.constant(round_keys, dtype=tf.int32)

    # load model
    start_time = time.time()
    cipher = tf.saved_model.load(model_path)
    model_load_time = time.time() - start_time
    res = cipher(plaintext, round_keys)
    inference_time = time.time() - start_time - model_load_time

    # print(cipher())
    print("tflite model outputs: ", res)
    print("\t model load time used:", model_load_time)
    print('\t inference time used:', inference_time)
    print(ciphertext)


if __name__ == '__main__':
    assert model_path != "YOUR_PATH", "Please specify the model path"

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # profile_data_movement()
        inference()
