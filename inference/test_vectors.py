#!/usr/bin/env python
# encoding: utf-8

import numpy as np

# All tests were taken from NIST, 2001 test vectors:
# https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38a.pdf

ecb_128 = {'key': '2b7e151628aed2a6abf7158809cf4f3c',
           'plaintext': ['6bc1bee22e409f96e93d7e117393172a', 'ae2d8a571e03ac9c9eb76fac45af8e51',
                         '30c81c46a35ce411e5fbc1191a0a52ef', 'f69f2445df4f9b17ad2b417be66c3710'],
           'ciphertext': ['3ad77bb40d7a3660a89ecaf32466ef97', 'f5d3d58503b9699de785895a96fdbaaf',
                          '43b1cd7f598ece23881b00e3ed030688', '7b0c785e27e8ad3f8223207104725dd4'],
           }


cbc_128 = {'key': '2b7e151628aed2a6abf7158809cf4f3c',
           'initial_vector': '000102030405060708090a0b0c0d0e0f',
           'plaintext': ['6bc1bee22e409f96e93d7e117393172a', 'ae2d8a571e03ac9c9eb76fac45af8e51',
                         '30c81c46a35ce411e5fbc1191a0a52ef', 'f69f2445df4f9b17ad2b417be66c3710'],
           'ciphertext': ['7649abac8119b246cee98e9b12e9197d', '5086cb9b507219ee95db113a917678b2',
                          '73bed6b8e3c1743b7116e69e22229516', '3ff1caa1681fac09120eca307586e1a7'],
            }


cfb_128 = {'key': '2b7e151628aed2a6abf7158809cf4f3c',
           'initial_vector': '000102030405060708090a0b0c0d0e0f',
           'plaintext': ['6bc1bee22e409f96e93d7e117393172a', 'ae2d8a571e03ac9c9eb76fac45af8e51',
                         '30c81c46a35ce411e5fbc1191a0a52ef', 'f69f2445df4f9b17ad2b417be66c3710'],
           'ciphertext': ['3b3fd92eb72dad20333449f8e83cfb4a', 'c8a64537a0b3a93fcde3cdad9f1ce58b',
                          '26751f67a3cbb140b1808cf187a4f4df', 'c04b05357c5d1c0eeac4c66f9ff7f2e6'],
            }


ofb_128 = {'key': '2b7e151628aed2a6abf7158809cf4f3c',
           'initial_vector': '000102030405060708090a0b0c0d0e0f',
           'plaintext': ['6bc1bee22e409f96e93d7e117393172a', 'ae2d8a571e03ac9c9eb76fac45af8e51',
                         '30c81c46a35ce411e5fbc1191a0a52ef', 'f69f2445df4f9b17ad2b417be66c3710'],
           'ciphertext': ['3b3fd92eb72dad20333449f8e83cfb4a', '7789508d16918f03f53c52dac54ed825',
                          '9740051e9c5fecf64344f7a82260edcc', '304c6528f659c77866a510d9c1d6ae5e'],
            }


ctr_128 = {'key': '2b7e151628aed2a6abf7158809cf4f3c',
           'initial_counter': 'f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff',
           'plaintext': ['6bc1bee22e409f96e93d7e117393172a', 'ae2d8a571e03ac9c9eb76fac45af8e51',
                         '30c81c46a35ce411e5fbc1191a0a52ef', 'f69f2445df4f9b17ad2b417be66c3710'],
           'ciphertext': ['874d6191b620e3261bef6864990db6ce', '9806f66b7970fdff8617187bb9fffdff',
                          '5ae4df3edbd5d35e5b4f09020db03eab', '1e031dda2fbe03d1792170a0f3009cee'],
            }


def get_vectors(mode, num_duplicate=1, single_block=False):
    iv, key, plaintext, ciphertext = '', '', '', ''
    if mode == 'ecb_128':
        key = ecb_128['key']
        plaintext = ''.join(ecb_128['plaintext'])
        ciphertext = ''.join(ecb_128['ciphertext'])
        if single_block:
            plaintext = plaintext[:32]
            ciphertext = ciphertext[:32]
        return key, plaintext * num_duplicate, ciphertext * num_duplicate
    elif mode == 'cbc_128':
        key = cbc_128['key']
        iv = cbc_128['initial_vector']
        plaintext = ''.join(cbc_128['plaintext'])
        ciphertext = ''.join(cbc_128['ciphertext'])
        if single_block:
            plaintext = plaintext[:32]
            ciphertext = ciphertext[:32]
        return key, iv, plaintext * num_duplicate, ciphertext * num_duplicate
    elif mode == 'cfb_128':
        key = cfb_128['key']
        iv = cfb_128['initial_vector']
        plaintext = ''.join(cfb_128['plaintext'])
        ciphertext = ''.join(cfb_128['ciphertext'])
        if single_block:
            plaintext = plaintext[:32]
            ciphertext = ciphertext[:32]
        return key, iv, plaintext * num_duplicate, ciphertext * num_duplicate
    elif mode == 'ofb_128':
        key = ofb_128['key']
        iv = ofb_128['initial_vector']
        plaintext = ''.join(ofb_128['plaintext'])
        ciphertext = ''.join(ofb_128['ciphertext'])
        if single_block:
            plaintext = plaintext[:32]
            ciphertext = ciphertext[:32]
        return key, iv, plaintext * num_duplicate, ciphertext * num_duplicate
    elif mode == 'ctr_128':
        key = ctr_128['key']
        iv = ctr_128['initial_counter']
        plaintext = ''.join(ctr_128['plaintext'])
        ciphertext = ''.join(ctr_128['ciphertext'])
        if single_block:
            plaintext = plaintext[:32]
            ciphertext = ciphertext[:32]
        return key, iv, plaintext * num_duplicate, ciphertext * num_duplicate
    else:
        return key, plaintext, ciphertext


def text2vector(hex_text):
    assert len(hex_text) % 2 == 0, 'length of hex_text should be even'
    res = []
    for i in range(int(len(hex_text) / 2)):
        b = hex_text[2*i: 2*i+2]
        b = '0x' + b
        b = int(b, 16)
        res.append(b)
    return res


def int_arr_to_bytes(int_arr):
    res = b''
    for p in int_arr:
        p = p.to_bytes(1, 'big')
        res = res + p
    res = bytes(res)
    return res


def write_bytes_to_file():
    key, plaintext, ciphertext = get_vectors('ecb_128')
    plaintext = text2vector(plaintext)
    ciphertext = text2vector(ciphertext)
    plaintext = int_arr_to_bytes(plaintext)
    ciphertext = int_arr_to_bytes(ciphertext)
    print(plaintext, len(plaintext))
    print(ciphertext, len(ciphertext))
    with open("/home/xin/Documents/code/c++/gpu-aes-whitebox/data/plaintext.txt", 'wb') as f:
        f.write(plaintext)
    with open("/home/xin/Documents/code/c++/gpu-aes-whitebox/data/ciphertext.txt", 'wb') as f:
        f.write(ciphertext)


def create_test_files():
    plaintext = '6bc1bee22e409f96e93d7e117393172a'
    plaintext = text2vector(plaintext)

    with open("data/CUDA_test_files.csv", 'a+') as f:
        print("block_num,file_name", file=f)

    for i in np.logspace(1, 8, num=15, dtype='int'):
        print(i)
        block_num = i

        # process plaintext
        p = plaintext * block_num

        p = int_arr_to_bytes(p)

        with open("/home/xin/Documents/code/c++/gpu-aes-whitebox/data/plaintext_{}.txt".format(block_num), 'wb') as f:
            f.write(p)
        with open("data/CUDA_test_files.csv", 'a+') as f:
            print("{},plaintext_{}.txt".format(block_num, block_num), file=f)

# write_bytes_to_file()
# create_test_files()