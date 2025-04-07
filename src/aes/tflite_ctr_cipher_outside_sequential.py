#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
from test_vectors import get_vectors, text2vector
from tflite_runtime.interpreter import Interpreter
from basic_block_parallel import key_expansion

model_path = "YOUR_PATH"
assert model_path != "YOUR_PATH", "Please specify the model path"

@tf.function(input_signature=[tf.TensorSpec(shape=(None, 16), dtype=tf.int32),
                                  tf.TensorSpec(shape=(None, 16), dtype=tf.int32),
                                  tf.TensorSpec(shape=(176,), dtype=tf.int32),
                              tf.TensorSpec(shape=(), dtype=tf.int32)],
                 experimental_compile=True)
def cipher(counter, plaintext, round_keys, length):
        Nr = 10
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

        gf_mul0 = [0x00, 0x02, 0x04, 0x06, 0x08, 0x0a, 0x0c, 0x0e, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e,
                   0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e,
                   0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x5a, 0x5c, 0x5e,
                   0x60, 0x62, 0x64, 0x66, 0x68, 0x6a, 0x6c, 0x6e, 0x70, 0x72, 0x74, 0x76, 0x78, 0x7a, 0x7c, 0x7e,
                   0x80, 0x82, 0x84, 0x86, 0x88, 0x8a, 0x8c, 0x8e, 0x90, 0x92, 0x94, 0x96, 0x98, 0x9a, 0x9c, 0x9e,
                   0xa0, 0xa2, 0xa4, 0xa6, 0xa8, 0xaa, 0xac, 0xae, 0xb0, 0xb2, 0xb4, 0xb6, 0xb8, 0xba, 0xbc, 0xbe,
                   0xc0, 0xc2, 0xc4, 0xc6, 0xc8, 0xca, 0xcc, 0xce, 0xd0, 0xd2, 0xd4, 0xd6, 0xd8, 0xda, 0xdc, 0xde,
                   0xe0, 0xe2, 0xe4, 0xe6, 0xe8, 0xea, 0xec, 0xee, 0xf0, 0xf2, 0xf4, 0xf6, 0xf8, 0xfa, 0xfc, 0xfe,
                   0x1b, 0x19, 0x1f, 0x1d, 0x13, 0x11, 0x17, 0x15, 0x0b, 0x09, 0x0f, 0x0d, 0x03, 0x01, 0x07, 0x05,
                   0x3b, 0x39, 0x3f, 0x3d, 0x33, 0x31, 0x37, 0x35, 0x2b, 0x29, 0x2f, 0x2d, 0x23, 0x21, 0x27, 0x25,
                   0x5b, 0x59, 0x5f, 0x5d, 0x53, 0x51, 0x57, 0x55, 0x4b, 0x49, 0x4f, 0x4d, 0x43, 0x41, 0x47, 0x45,
                   0x7b, 0x79, 0x7f, 0x7d, 0x73, 0x71, 0x77, 0x75, 0x6b, 0x69, 0x6f, 0x6d, 0x63, 0x61, 0x67, 0x65,
                   0x9b, 0x99, 0x9f, 0x9d, 0x93, 0x91, 0x97, 0x95, 0x8b, 0x89, 0x8f, 0x8d, 0x83, 0x81, 0x87, 0x85,
                   0xbb, 0xb9, 0xbf, 0xbd, 0xb3, 0xb1, 0xb7, 0xb5, 0xab, 0xa9, 0xaf, 0xad, 0xa3, 0xa1, 0xa7, 0xa5,
                   0xdb, 0xd9, 0xdf, 0xdd, 0xd3, 0xd1, 0xd7, 0xd5, 0xcb, 0xc9, 0xcf, 0xcd, 0xc3, 0xc1, 0xc7, 0xc5,
                   0xfb, 0xf9, 0xff, 0xfd, 0xf3, 0xf1, 0xf7, 0xf5, 0xeb, 0xe9, 0xef, 0xed, 0xe3, 0xe1, 0xe7, 0xe5,
                   ]

        gf_mul1 = [0x00, 0x03, 0x06, 0x05, 0x0c, 0x0f, 0x0a, 0x09, 0x18, 0x1b, 0x1e, 0x1d, 0x14, 0x17, 0x12, 0x11,
                   0x30, 0x33, 0x36, 0x35, 0x3c, 0x3f, 0x3a, 0x39, 0x28, 0x2b, 0x2e, 0x2d, 0x24, 0x27, 0x22, 0x21,
                   0x60, 0x63, 0x66, 0x65, 0x6c, 0x6f, 0x6a, 0x69, 0x78, 0x7b, 0x7e, 0x7d, 0x74, 0x77, 0x72, 0x71,
                   0x50, 0x53, 0x56, 0x55, 0x5c, 0x5f, 0x5a, 0x59, 0x48, 0x4b, 0x4e, 0x4d, 0x44, 0x47, 0x42, 0x41,
                   0xc0, 0xc3, 0xc6, 0xc5, 0xcc, 0xcf, 0xca, 0xc9, 0xd8, 0xdb, 0xde, 0xdd, 0xd4, 0xd7, 0xd2, 0xd1,
                   0xf0, 0xf3, 0xf6, 0xf5, 0xfc, 0xff, 0xfa, 0xf9, 0xe8, 0xeb, 0xee, 0xed, 0xe4, 0xe7, 0xe2, 0xe1,
                   0xa0, 0xa3, 0xa6, 0xa5, 0xac, 0xaf, 0xaa, 0xa9, 0xb8, 0xbb, 0xbe, 0xbd, 0xb4, 0xb7, 0xb2, 0xb1,
                   0x90, 0x93, 0x96, 0x95, 0x9c, 0x9f, 0x9a, 0x99, 0x88, 0x8b, 0x8e, 0x8d, 0x84, 0x87, 0x82, 0x81,
                   0x9b, 0x98, 0x9d, 0x9e, 0x97, 0x94, 0x91, 0x92, 0x83, 0x80, 0x85, 0x86, 0x8f, 0x8c, 0x89, 0x8a,
                   0xab, 0xa8, 0xad, 0xae, 0xa7, 0xa4, 0xa1, 0xa2, 0xb3, 0xb0, 0xb5, 0xb6, 0xbf, 0xbc, 0xb9, 0xba,
                   0xfb, 0xf8, 0xfd, 0xfe, 0xf7, 0xf4, 0xf1, 0xf2, 0xe3, 0xe0, 0xe5, 0xe6, 0xef, 0xec, 0xe9, 0xea,
                   0xcb, 0xc8, 0xcd, 0xce, 0xc7, 0xc4, 0xc1, 0xc2, 0xd3, 0xd0, 0xd5, 0xd6, 0xdf, 0xdc, 0xd9, 0xda,
                   0x5b, 0x58, 0x5d, 0x5e, 0x57, 0x54, 0x51, 0x52, 0x43, 0x40, 0x45, 0x46, 0x4f, 0x4c, 0x49, 0x4a,
                   0x6b, 0x68, 0x6d, 0x6e, 0x67, 0x64, 0x61, 0x62, 0x73, 0x70, 0x75, 0x76, 0x7f, 0x7c, 0x79, 0x7a,
                   0x3b, 0x38, 0x3d, 0x3e, 0x37, 0x34, 0x31, 0x32, 0x23, 0x20, 0x25, 0x26, 0x2f, 0x2c, 0x29, 0x2a,
                   0x0b, 0x08, 0x0d, 0x0e, 0x07, 0x04, 0x01, 0x02, 0x13, 0x10, 0x15, 0x16, 0x1f, 0x1c, 0x19, 0x1a,
                   ]

        def shift_rows(input_text):
            shift_table = tf.constant([
                0, 5, 10, 15,
                4, 9, 14, 3,
                8, 13, 2, 7,
                12, 1, 6, 11,
            ], dtype=tf.int64)
            res = tf.gather(input_text, indices=shift_table, axis=1)
            return res

        def sbox_lookup(state):
            state = tf.gather(sbox, indices=state)
            return state

        def mix_column(state):
            # 0-3
            aa = tf.gather(state, indices=[0], axis=1)
            bb = tf.gather(state, indices=[1], axis=1)
            cc = tf.gather(state, indices=[2], axis=1)
            dd = tf.gather(state, indices=[3], axis=1)

            # state[0]
            a0 = tf.gather(gf_mul0, aa)
            b1 = tf.gather(gf_mul1, bb)
            ab = tf.bitwise.bitwise_xor(a0, b1)
            cd = tf.bitwise.bitwise_xor(cc, dd)
            state0 = tf.bitwise.bitwise_xor(ab, cd)

            # state[1]
            b0 = tf.gather(gf_mul0, bb)
            c1 = tf.gather(gf_mul1, cc)
            ab = tf.bitwise.bitwise_xor(aa, b0)
            cd = tf.bitwise.bitwise_xor(c1, dd)
            state1 = tf.bitwise.bitwise_xor(ab, cd)

            # state[2]
            c0 = tf.gather(gf_mul0, cc)
            d1 = tf.gather(gf_mul1, dd)
            ab = tf.bitwise.bitwise_xor(aa, bb)
            cd = tf.bitwise.bitwise_xor(c0, d1)
            state2 = tf.bitwise.bitwise_xor(ab, cd)

            # state[3]
            a1 = tf.gather(gf_mul1, aa)
            d0 = tf.gather(gf_mul0, dd)
            ab = tf.bitwise.bitwise_xor(a1, bb)
            cd = tf.bitwise.bitwise_xor(cc, d0)
            state3 = tf.bitwise.bitwise_xor(ab, cd)

            # return state3

            # 4-7
            aa = tf.gather(state, indices=[4], axis=1)
            bb = tf.gather(state, indices=[5], axis=1)
            cc = tf.gather(state, indices=[6], axis=1)
            dd = tf.gather(state, indices=[7], axis=1)

            # state[4]
            a0 = tf.gather(gf_mul0, aa)
            b1 = tf.gather(gf_mul1, bb)
            ab = tf.bitwise.bitwise_xor(a0, b1)
            cd = tf.bitwise.bitwise_xor(cc, dd)
            state4 = tf.bitwise.bitwise_xor(ab, cd)

            # state[5]
            b0 = tf.gather(gf_mul0, bb)
            c1 = tf.gather(gf_mul1, cc)
            ab = tf.bitwise.bitwise_xor(aa, b0)
            cd = tf.bitwise.bitwise_xor(c1, dd)
            state5 = tf.bitwise.bitwise_xor(ab, cd)

            # state[6]
            c0 = tf.gather(gf_mul0, cc)
            d1 = tf.gather(gf_mul1, dd)
            ab = tf.bitwise.bitwise_xor(aa, bb)
            cd = tf.bitwise.bitwise_xor(c0, d1)
            state6 = tf.bitwise.bitwise_xor(ab, cd)

            # state[7]
            a1 = tf.gather(gf_mul1, aa)
            d0 = tf.gather(gf_mul0, dd)
            ab = tf.bitwise.bitwise_xor(a1, bb)
            cd = tf.bitwise.bitwise_xor(cc, d0)
            state7 = tf.bitwise.bitwise_xor(ab, cd)

            # 8-11
            aa = tf.gather(state, indices=[8], axis=1)
            bb = tf.gather(state, indices=[9], axis=1)
            cc = tf.gather(state, indices=[10], axis=1)
            dd = tf.gather(state, indices=[11], axis=1)

            # state[8]
            a0 = tf.gather(gf_mul0, aa)
            b1 = tf.gather(gf_mul1, bb)
            ab = tf.bitwise.bitwise_xor(a0, b1)
            cd = tf.bitwise.bitwise_xor(cc, dd)
            state8 = tf.bitwise.bitwise_xor(ab, cd)

            # state[9]
            b0 = tf.gather(gf_mul0, bb)
            c1 = tf.gather(gf_mul1, cc)
            ab = tf.bitwise.bitwise_xor(aa, b0)
            cd = tf.bitwise.bitwise_xor(c1, dd)
            state9 = tf.bitwise.bitwise_xor(ab, cd)

            # state[10]
            c0 = tf.gather(gf_mul0, cc)
            d1 = tf.gather(gf_mul1, dd)
            ab = tf.bitwise.bitwise_xor(aa, bb)
            cd = tf.bitwise.bitwise_xor(c0, d1)
            state10 = tf.bitwise.bitwise_xor(ab, cd)

            # state[11]
            a1 = tf.gather(gf_mul1, aa)
            d0 = tf.gather(gf_mul0, dd)
            ab = tf.bitwise.bitwise_xor(a1, bb)
            cd = tf.bitwise.bitwise_xor(cc, d0)
            state11 = tf.bitwise.bitwise_xor(ab, cd)

            # 12-15
            aa = tf.gather(state, indices=[12], axis=1)
            bb = tf.gather(state, indices=[13], axis=1)
            cc = tf.gather(state, indices=[14], axis=1)
            dd = tf.gather(state, indices=[15], axis=1)

            # state[12]
            a0 = tf.gather(gf_mul0, aa)
            b1 = tf.gather(gf_mul1, bb)
            ab = tf.bitwise.bitwise_xor(a0, b1)
            cd = tf.bitwise.bitwise_xor(cc, dd)
            state12 = tf.bitwise.bitwise_xor(ab, cd)

            # state[13]
            b0 = tf.gather(gf_mul0, bb)
            c1 = tf.gather(gf_mul1, cc)
            ab = tf.bitwise.bitwise_xor(aa, b0)
            cd = tf.bitwise.bitwise_xor(c1, dd)
            state13 = tf.bitwise.bitwise_xor(ab, cd)

            # state[14]
            c0 = tf.gather(gf_mul0, cc)
            d1 = tf.gather(gf_mul1, dd)
            ab = tf.bitwise.bitwise_xor(aa, bb)
            cd = tf.bitwise.bitwise_xor(c0, d1)
            state14 = tf.bitwise.bitwise_xor(ab, cd)

            # state[15]
            a1 = tf.gather(gf_mul1, aa)
            d0 = tf.gather(gf_mul0, dd)
            ab = tf.bitwise.bitwise_xor(a1, bb)
            cd = tf.bitwise.bitwise_xor(cc, d0)
            state15 = tf.bitwise.bitwise_xor(ab, cd)

            state = tf.concat(values=[state0, state1, state2, state3,
                                      state4, state5, state6, state7,
                                      state8, state9, state10, state11,
                                      state12, state13, state14, state15], axis=1)
            return state

        def add_round_key(state, key):
            state = tf.bitwise.bitwise_xor(state, key)
            return state

        def cipher(plaintext, round_key):

            key = tf.slice(round_key, begin=[0], size=[16])
            state = add_round_key(plaintext, key)

            for i in range(Nr - 1):
                state = sbox_lookup(state)
                state = shift_rows(state)
                state = mix_column(state)
                key = tf.slice(round_key, begin=[(i + 1) * 16], size=[16])
                state = add_round_key(state, key)
            state = sbox_lookup(state)
            state = shift_rows(state)
            key = tf.slice(round_key, begin=[10 * 16], size=[16])
            state = add_round_key(state, key)
            return state

        ciphertext = plaintext

        for i in range(length):
            block = tf.gather(counter, indices=[i])
            text = cipher(block, round_keys)
            text = tf.cast(text, dtype=tf.int32)
            p = tf.gather(plaintext, indices=[i])
            # p = tf.reshape()
            text = tf.bitwise.bitwise_xor(text, p)
            indices = [[i]]
            updates = tf.reshape(text, shape=(1, 16))
            ciphertext = tf.tensor_scatter_nd_update(ciphertext, indices, updates)

        return ciphertext

        # return ciphertext

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

def save_model():
    _, counter, plaintext, ciphertext = get_vectors('ctr_128', single_block=True)

    block_num = int(len(plaintext) / 32)

    counter = text2vector(counter)
    counter = counter_generator(counter, block_num)
    # length = tf.constant(block_num, dtype=tf.int32)
    counter = tf.constant(counter, dtype=tf.int32)
    # counter = tf.constant(counter, dtype=tf.int32)
    counter = tf.reshape(counter, shape=(block_num, 16))

    # process length
    length = tf.constant(block_num, dtype=tf.int32)

    key = '2b7e151628aed2a6abf7158809cf4f3c'
    key = text2vector(key)
    round_keys = key_expansion(key)
    print(round_keys)
    round_keys = tf.constant(round_keys, dtype=tf.int32)

    plaintext = text2vector(plaintext)
    plaintext = tf.constant(plaintext, dtype=tf.int32)
    plaintext = tf.reshape(plaintext, shape=(block_num, 16))

    concrete_func = cipher.get_concrete_function(counter, plaintext, round_keys, length)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    # Save the model.
    with open(model_path, 'wb') as f:
        f.write(tflite_model)


def tflite_inference():
    _, counter, plaintext, ciphertext = get_vectors('ctr_128', single_block=True)

    block_num = int(len(plaintext) / 32)

    # process initial vector
    counter = text2vector(counter)
    counter = counter_generator(counter, block_num)
    # length = tf.constant(block_num, dtype=tf.int32)
    counter = tf.constant(counter, dtype=tf.int32)
    # counter = tf.constant(counter, dtype=tf.int32)
    counter = tf.reshape(counter, shape=(block_num, 16))
    # process length
    length = tf.constant(block_num, dtype=tf.int32)

    # process plaintext
    plaintext = text2vector(plaintext)
    plaintext = tf.constant(plaintext, dtype=tf.int32)
    plaintext = tf.reshape(plaintext, shape=(block_num, 16))

    key = '2b7e151628aed2a6abf7158809cf4f3c'
    key = text2vector(key)
    round_keys = key_expansion(key)
    print(round_keys)
    round_keys = tf.constant(round_keys, dtype=tf.int32)

    tflite_model_path = model_path

    interpreter = Interpreter(model_path=tflite_model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(input_details[0]['index'], (block_num, 16))
    interpreter.resize_tensor_input(input_details[1]['index'], (block_num, 16))
    interpreter.allocate_tensors()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], counter)
    interpreter.set_tensor(input_details[1]['index'], plaintext)
    interpreter.set_tensor(input_details[2]['index'],round_keys)
    interpreter.set_tensor(input_details[3]['index'], length)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)

if __name__ == '__main__':
    # simple_test()
    # save_model()
    tflite_inference()