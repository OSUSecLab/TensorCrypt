#!/usr/bin/env python
# encoding: utf-8

from chacha20_cipher import Cipher
from chacha20_cipher import counter_generator
import tensorflow as tf
import numpy as np
from tflite_runtime.interpreter import Interpreter

def save_model():
    key = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    nonce = [101, 102, 103, 104, 105, 106, 107, 108]
    
    counter_input = counter_generator(key, nonce, block_number=4)
    keystream = tf.constant(counter_input, dtype=tf.int32)
    plaintext = tf.zeros(shape=(4, 64), dtype=tf.int32)
    
    cipher = Cipher()
    concrete_func = cipher.__call__.get_concrete_function(keystream, plaintext)
    
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    
    # Save the model.
    with open('data/chacha20_cipher.tflite', 'wb') as f:
        f.write(tflite_model)
        
def tflite_inference():
    key = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    nonce = [101, 102, 103, 104, 105, 106, 107, 108]
    block_num = 4
    
    counter_input = counter_generator(key, nonce, block_number=4)
    keystream = tf.constant(counter_input, dtype=tf.int32)
    plaintext = tf.zeros(shape=(4, 64), dtype=tf.int32)

    tflite_model_path = 'data/chacha20_cipher.tflite'

    interpreter = Interpreter(model_path=tflite_model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(input_details[0]['index'], (block_num, 64))
    interpreter.resize_tensor_input(input_details[1]['index'], (block_num, 64))
    interpreter.allocate_tensors()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], keystream)
    interpreter.set_tensor(input_details[1]['index'], plaintext)
    # interpreter.set_tensor(input_details[2]['index'],round_keys)
    # interpreter.set_tensor(input_details[3]['index'], length)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
        
if __name__ == '__main__':
    save_model()
    tflite_inference()