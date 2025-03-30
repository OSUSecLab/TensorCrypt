#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from salsa20_parallel_test import counter_generator

model_path = "YOUR_PATH"

def inference():
    cipher = tf.saved_model.load(model_path)
    # print(cipher(plaintext))
    key = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    nonce = [101, 102, 103, 104, 105, 106, 107, 108]
    counter_input = counter_generator(key, nonce, block_number=4)
    keystream = tf.constant(counter_input, dtype=tf.int32)
    plaintext = tf.zeros(shape=(4, 64), dtype=tf.int32)

    ciphertext = cipher(keystream, plaintext)
    ciphertext = cipher(keystream, ciphertext)
    print(ciphertext)

if __name__ == '__main__':
    assert model_path != "YOUR_PATH", "Please specify the model path"
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        save_to_model()
        inference()