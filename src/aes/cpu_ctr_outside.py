#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import time
import tensorflow as tf
import datetime
from test_vectors import get_vectors, text2vector

def key_expansion(key):
    nk, nb = 4, 4
    nr = 10
    rcon = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
            0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
            0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
            0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39]

    sbox = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
    ]

    round_keys = []
    for i in range(16):
        if i % 4 == 0:
            round_keys.append([key[i]])
        else:
            round_keys[i // 4].append(key[i])
    # print round_keys

    for i in range(4, 4 * 11):
        round_keys.append([])
        if i % 4 == 0:
            byte = round_keys[i - 4][0] \
                   ^ sbox[round_keys[i - 1][1]] \
                   ^ rcon[i // 4]
            round_keys[i].append(byte)

            for j in range(1, 4):
                byte = round_keys[i - 4][j] \
                       ^ sbox[round_keys[i - 1][(j + 1) % 4]]
                round_keys[i].append(byte)
        else:
            for j in range(4):
                byte = round_keys[i - 4][j] \
                       ^ round_keys[i - 1][j]
                round_keys[i].append(byte)
    result = []
    for key in round_keys:
        result += key
    # return round_keys
    return result

def counter_generator(counter, block_number):
    counter = np.asarray(counter, dtype=np.uint8)
    tmp = np.zeros(shape=(block_number, 1), dtype=np.uint8)
    counter_input = np.add(counter, tmp)
    counter_quotient = np.zeros(shape=(block_number,), dtype=np.int64)
    offset = np.arange(0, block_number, dtype=np.int64)
    for j in range(15, -1, -1):
        col = counter_input[:,j]
        if j == 15:
            col = np.add(col, offset)
        col = np.add(col, counter_quotient)
        counter_input[:,j] = col.astype(np.uint8)
        counter_quotient = np.right_shift(col, 8)
    return counter_input

def ctr_mode_test():
    now = datetime.datetime.now()
    cur_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    model_path = "data/ctr_chipher_counter_outside"
    _, counter, plaintext, ciphertext = get_vectors('ctr_128', single_block=True)

    # block_num = int(len(plaintext) / 32)
    plaintext = text2vector(plaintext)
    counter = text2vector(counter)

    key = '2b7e151628aed2a6abf7158809cf4f3c'
    key = text2vector(key)
    start_time = time.time()
    round_keys = key_expansion(key)
    key_expansion_time = time.time() - start_time

    csv_path = f"data/ctr_outside_cpu_time.csv"
    with open(csv_path, 'a+') as f:
        print("---- Current time:", cur_time, "----", file=f)
        print("block_num,key_expansion,model_load_time,first_time_sec,load_plaintext_tensor,GPU_and_retrieval_time", file=f)

    start_time = time.time()
    cipher = tf.saved_model.load(model_path)
    model_load_time = time.time() - start_time
    print("Model load time used: %f sec" % model_load_time)

    for k, i in enumerate(np.logspace(1, 8, num=15, dtype='int')):
        if i > 10000000:
            break
        block_num = i
        p = plaintext * block_num
        start_time = time.time()
        round_keys = tf.constant(round_keys, dtype=tf.int32)
        p = tf.constant(p, dtype=tf.int32, shape=(block_num, 16))
        ctr = counter_generator(counter, block_num)
        ctr = tf.constant(ctr, dtype=tf.int32, shape=(block_num, 16))
        # length = tf.constant(block_num, dtype=tf.int32)
        load_tensor_to_GPU = time.time() - start_time
        print("block:", block_num)

        total_time = 0
        num_iter = 10
        start_time = time.time()
        res = cipher(ctr, p, round_keys)
        first_time_sec = time.time() - start_time

        for j in range(num_iter):

            start_time = time.time()
            res = cipher(ctr, p, round_keys)
            inference_time = time.time() - start_time
            total_time += inference_time
            print("iter:", j, inference_time)
        average_time = total_time / num_iter
        with open(csv_path, 'a+') as f:
            print("{},{},{},{},{},{}".format(block_num, key_expansion_time, model_load_time, first_time_sec,
                                             load_tensor_to_GPU, average_time), file=f)


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        ctr_mode_test()