#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf

model_path = "YOUR_PATH"

def generate_array(seed_arr, block_number):
    block_size = len(seed_arr)
    array = np.zeros(shape=(block_number, block_size), dtype=np.uint8)
    array = np.add(array, np.array(seed_arr, dtype=np.uint8))
    return array

def counter_generator(key, nonce, block_number, block_size=4):
    # counter = np.asarray(counter, dtype=np.uint8)
    counter_input = np.zeros(shape=(block_number, block_size), dtype=np.uint8)
    # counter_input = np.add(counter, tmp)
    part1 = generate_array(seed_arr=[101, 120, 112, 97, 110, 100, 32, 49, 54, 45, 98, 121, 116, 101, 32, 107] + key + key
                           , block_number=block_number)
    # part1 = generate_array(seed_arr=[101, 120, 112, 97] + key + [110, 100, 32, 49]+nonce, block_number=block_number)
    part2 = generate_array(seed_arr=[0, 0, 0, 0] + nonce, block_number=block_number)
    counter_quotient = np.zeros(shape=(block_number,), dtype=np.int64)
    offset = np.arange(0, block_number, dtype=np.int64)
    # for j in range(block_size-1, -1, -1):
    for j in range(block_size):
        col = counter_input[:,j]
        if j == 0:
            col = np.add(col, offset)
        col = np.add(col, counter_quotient)
        counter_input[:,j] = col.astype(np.uint8)
        counter_quotient = np.right_shift(col, 8)

    res = np.concatenate((part1, counter_input, part2), axis=1)
    return res



class Cipher(tf.Module):

    def __init__(self):
        super(Cipher, self).__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 64), dtype=tf.int32),
                                  tf.TensorSpec(shape=(None, 64), dtype=tf.int32)], experimental_compile=True)
    def __call__(
            self,
            keystream,
            plaintext,
    ):
        def rotl(value, shift):
            aa = tf.bitwise.left_shift(value, shift)
            aa = tf.bitwise.bitwise_and(aa, 0xffffffff)
            bb = tf.bitwise.right_shift(value, 32 - shift)
            return tf.bitwise.bitwise_or(aa, bb)

        def c20_quarterround(y0, y1, y2, y3):
            y0 = tf.bitwise.bitwise_and(tf.math.add(y0, y1), 0xffffffff)
            y3 = rotl(tf.bitwise.bitwise_xor(y3, y0), 16)

            y2 = tf.bitwise.bitwise_and(tf.math.add(y2, y3), 0xffffffff)
            y1 = rotl(tf.bitwise.bitwise_xor(y1, y2), 12)

            y0 = tf.bitwise.bitwise_and(tf.math.add(y0, y1), 0xffffffff)
            y3 = rotl(tf.bitwise.bitwise_xor(y3, y0), 8)

            y2 = tf.bitwise.bitwise_and(tf.math.add(y2, y3), 0xffffffff)
            y1 = rotl(tf.bitwise.bitwise_xor(y1, y2), 7)
            return y0, y1, y2, y3

        def c20_column_round(stream):
            x0 = tf.gather(stream, indices=[0], axis=1)
            x4 = tf.gather(stream, indices=[4], axis=1)
            x8 = tf.gather(stream, indices=[8], axis=1)
            x12 = tf.gather(stream, indices=[12], axis=1)
            x0, x4, x8, x12 = c20_quarterround(x0, x4, x8, x12)

            x1 = tf.gather(stream, indices=[1], axis=1)
            x5 = tf.gather(stream, indices=[5], axis=1)
            x9 = tf.gather(stream, indices=[9], axis=1)
            x13 = tf.gather(stream, indices=[13], axis=1)
            x1, x5, x9, x13 = c20_quarterround(x1, x5, x9, x13)

            x2 = tf.gather(stream, indices=[2], axis=1)
            x6 = tf.gather(stream, indices=[6], axis=1)
            x10 = tf.gather(stream, indices=[10], axis=1)
            x14 = tf.gather(stream, indices=[14], axis=1)
            x2, x6, x10, x14 = c20_quarterround(x2, x6, x10, x14)

            x3 = tf.gather(stream, indices=[3], axis=1)
            x7 = tf.gather(stream, indices=[7], axis=1)
            x11 = tf.gather(stream, indices=[11], axis=1)
            x15 = tf.gather(stream, indices=[15], axis=1)
            x3, x7, x11, x15 = c20_quarterround(x3, x7, x11, x15)

            x = tf.concat(values=[x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15], axis=1)
            return x

        def c20_diagonal(stream):
            x0 = tf.gather(stream, indices=[0], axis=1)
            x5 = tf.gather(stream, indices=[5], axis=1)
            x10 = tf.gather(stream, indices=[10], axis=1)
            x15 = tf.gather(stream, indices=[15], axis=1)
            x0, x5, x10, x15 = c20_quarterround(x0, x5, x10, x15)

            x1 = tf.gather(stream, indices=[1], axis=1)
            x6 = tf.gather(stream, indices=[6], axis=1)
            x11 = tf.gather(stream, indices=[11], axis=1)
            x12 = tf.gather(stream, indices=[12], axis=1)
            x1, x6, x11, x12 = c20_quarterround(x1, x6, x11, x12)

            x2 = tf.gather(stream, indices=[2], axis=1)
            x7 = tf.gather(stream, indices=[7], axis=1)
            x8 = tf.gather(stream, indices=[8], axis=1)
            x13 = tf.gather(stream, indices=[13], axis=1)
            x2, x7, x8, x13 = c20_quarterround(x2, x7, x8, x13)

            x3 = tf.gather(stream, indices=[3], axis=1)
            x4 = tf.gather(stream, indices=[4], axis=1)
            x9 = tf.gather(stream, indices=[9], axis=1)
            x14 = tf.gather(stream, indices=[14], axis=1)
            x3, x4, x9, x14 = c20_quarterround(x3, x4, x9, x14)

            x = tf.concat(values=[x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15], axis=1)
            return x

        def c20_double_round(stream):
            x = c20_column_round(stream)
            x = c20_diagonal(x)
            return x

        def c20_hash(stream):
            aa = tf.gather(stream, indices=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60], axis=1)
            bb = tf.gather(stream, indices=[1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61], axis=1)
            cc = tf.gather(stream, indices=[2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62], axis=1)
            dd = tf.gather(stream, indices=[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63], axis=1)

            bb = tf.bitwise.left_shift(bb, 8)
            bb = tf.bitwise.bitwise_and(bb, 0xffffffff)

            cc = tf.bitwise.left_shift(cc, 16)
            cc = tf.bitwise.bitwise_and(cc, 0xffffffff)

            dd = tf.bitwise.left_shift(dd, 24)
            dd = tf.bitwise.bitwise_and(dd, 0xffffffff)

            x = tf.math.add(aa, bb)
            x = tf.math.add(x, cc)
            x = tf.math.add(x, dd)

            z = x

            for i in range(10):
                z = c20_double_round(z)

            z = tf.math.add(z, x)
            z0 = tf.bitwise.bitwise_and(z, 0xff)
            z1 = tf.bitwise.bitwise_and(tf.bitwise.right_shift(z, 8), 0xff)
            z2 = tf.bitwise.bitwise_and(tf.bitwise.right_shift(z, 16), 0xff)
            z3 = tf.bitwise.bitwise_and(tf.bitwise.right_shift(z, 24), 0xff)

            z = tf.concat(values=[z0, z1, z2, z3], axis=1)

            keystream = tf.gather(z, indices=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
                                              1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61,
                                              2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62,
                                              3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63], axis=1)
            return keystream

        hash_out = c20_hash(keystream)

        ciphertext = tf.bitwise.bitwise_xor(plaintext, hash_out)

        return ciphertext

def save_to_model():

    cipher = Cipher()
    # print(module(10))
    tf.saved_model.save(cipher, model_path)

def inference():
    cipher = Cipher()
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
    assert model_path != "YOUR_PATH", "Please specify the model path"

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        save_to_model()
        inference()
