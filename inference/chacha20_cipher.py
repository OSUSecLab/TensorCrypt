#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf

model_path = "YOUR_PATH"

def inference():
    cipher = tf.saved_model.load(model_path)
    # print(cipher(plaintext))
    k = [196, 110, 193, 177, 140, 232, 168, 120, 114, 90, 55, 231, 128, 223, 183, 53]
    n = [26, 218, 49, 213, 207, 104, 130, 33]
    counter_input = counter_generator(k, n, block_number=2)
    keystream = tf.constant(counter_input, dtype=tf.int32)
    plaintext = [84, 101, 115, 116, 84, 101, 115, 116, 84, 101, 115, 116, 84, 101,
                115, 116, 84, 101, 115, 116, 84, 101, 115, 116, 84, 101, 115, 116, 84, 101, 115,
                116, 84, 101, 115, 116, 84, 101, 115, 116, 84, 101, 115, 116, 84, 101, 115, 116,
                84, 101, 115, 116, 84, 101, 115, 116, 84, 101, 115, 116, 84, 101, 115, 116, 84,
                101, 115, 116, 84, 101, 115, 116, 84, 101, 115, 116, 84, 101, 115, 116, 84, 101,
                115, 116, 84, 101, 115, 116, 84, 101, 115, 116, 84, 101, 115, 116, 84, 101, 115,
                116, 84, 101, 115, 116, 84, 101, 115, 116, 84, 101, 115, 116, 84, 101, 115, 116,
                 84, 101, 115, 116, 84, 101, 115, 116, 84, 101, 115, 116]
    plaintext = tf.constant(plaintext, dtype=tf.int32)
    plaintext = tf.reshape(plaintext, [2, 64])

    expected = [214, 15, 206, 172, 16, 5, 145, 157, 96, 250, 125, 128, 251, 62, 100, 239,
                22, 11, 56, 89, 68, 255, 239, 47, 224, 37, 115, 218, 5, 219, 218, 126, 29, 14, 157,
                155, 54, 194, 27, 36, 171, 90, 119, 118, 144, 184, 186, 235, 57, 213, 12, 97, 72,
                121, 126, 142, 150, 128, 22, 17, 130, 77, 229, 81, 15, 70, 96, 90, 47, 35, 239, 15,
                175, 237, 137, 225, 128, 41, 214, 218, 106, 32, 155, 60, 240, 117, 253, 236, 238,
                178, 218, 159, 65, 52, 84, 240, 242, 204, 149, 145, 197, 185, 149, 0, 70, 111, 185,
                141, 80, 106, 134, 123, 167, 201, 67, 184, 175, 59, 103, 84, 27, 59, 49, 28, 237, 111, 251, 112]
    expected = tf.constant(expected, dtype=tf.int32)
    expected = tf.reshape(expected, [2, 64])
    ciphertext = cipher(keystream, plaintext)
    # ciphertext = cipher(keystream, ciphertext)
    # print(ciphertext)
    print("result comparison:", tf.reduce_all(tf.equal(ciphertext, expected)))


if __name__ == '__main__':
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inference()
