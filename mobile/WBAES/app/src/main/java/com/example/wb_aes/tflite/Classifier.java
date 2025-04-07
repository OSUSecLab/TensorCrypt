package com.example.wb_aes.tflite;

import android.app.Activity;
import android.content.Context;
import android.os.SystemClock;
import android.util.Log;

import java.io.File;
import java.io.IOError;
import java.io.IOException;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
//import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;


import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Classifier {
    private final String TAG = "Classifier";
    private MappedByteBuffer tfliteModel;
    private GpuDelegate gpuDelegate = null;
    protected Interpreter tflite;
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    private int numThreads;
    private int[] range = {10, 31, 100, 316, 1000, 3162, 10000, 31622, 100000, 316227, 1000000};
//    private int[] range = {31622, 100000};

    public enum Device {
        CPU,
        GPU
    }

    public Classifier(Activity activity, Device device, int numThreads, String modelName) throws IOException {
        tfliteModel = FileUtil.loadMappedFile(activity, getModelPath(modelName));
//        CompatibilityList compatList = new CompatibilityList();
//        Log.d(TAG, "compatList.isDelegateSupportedOnThisDevice()=" + compatList.isDelegateSupportedOnThisDevice());
        switch (device) {
            case GPU:
                // TODO: Create a GPU delegate instance and add it to the interpreter options
                gpuDelegate = new GpuDelegate();
                tfliteOptions.addDelegate(gpuDelegate);
                break;
            case CPU:
                break;
        }
        this.setNumThreads(numThreads);

        tfliteOptions.setNumThreads(numThreads);
//        NnApiDelegate nnApiDelegate = new NnApiDelegate();
//        tfliteOptions.addDelegate(nnApiDelegate);
//        tfliteOptions.setAllowBufferHandleOutput(true);
//        tfliteOptions.setUseNNAPI(true);
//        tfliteOptions.setUseXNNPACK(true);
        long startTimeForReference = SystemClock.uptimeMillis();
        tflite = new Interpreter(tfliteModel, tfliteOptions);
        long endTimeForReference = SystemClock.uptimeMillis();
        long model_load_time = (endTimeForReference - startTimeForReference);
        Log.d(TAG, modelName + " model load:" + model_load_time);
//        tflite.resizeInput();
    }

    public void resizeInput(int numBlocks) {
//        Log.d()
        Log.d(TAG, "Before resizing, input shape:" + Arrays.toString(tflite.getInputTensor(0).shape()));
        tflite.resizeInput(0, new int[]{numBlocks, 16});
        Log.d(TAG, "After resizing, input shape:" + Arrays.toString(tflite.getInputTensor(0).shape()));
    }

    public int[][] generate_input(int numBlocks){
        int[] seed = {107, 193, 190, 226, 46, 64, 159, 150, 233, 61, 126, 17, 115, 147, 23, 42};
        int[][] res = new int[numBlocks][16];
        for (int i = 0; i < numBlocks; i++) {
            System.arraycopy(seed, 0, res[i], 0, 16);
        }
        return res;
    }

    public void inference_array_test() {
//        int[] range = {10, 31, 100, 316, 1000, 3162, 10000, 31622, 100000, 316227, 1000000, 3162277, 10000000, 31622776, 100000000};
        int[] range = {10, 31, 100, 316, 1000, 3162, 10000, 31622, 100000, 316227, 1000000};
        for (int i = 0; i < range.length; i++) {

            int numBlocks = range[i];
            int[][] ciphertext = new int[numBlocks][16];
            long startTimeForReference = SystemClock.uptimeMillis();
            int[][] plaintext = generate_input(numBlocks);
            long endTimeForReference = SystemClock.uptimeMillis();
            long input_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            this.tflite.run(plaintext, ciphertext);
            endTimeForReference = SystemClock.uptimeMillis();
            long inference_time = (endTimeForReference - startTimeForReference);
            Log.d(TAG, numBlocks + ": " + input_generation_time + ", " + inference_time);
            plaintext = null;
            ciphertext = null;
            System.gc();
        }
    }

    private int[] generate_buffer_input(int numBlocks) {
        int[] seed = {107, 193, 190, 226, 46, 64, 159, 150, 233, 61, 126, 17, 115, 147, 23, 42};
        int[] res = new int[numBlocks * 16];
        for (int i = 0; i < numBlocks; i++) {
            System.arraycopy(seed, 0, res, i*16, 16);
        }
        return res;
    }

    private int[] chacha_generate_buffer_input(int numBlocks) {
        int[] seed = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int[] res = new int[numBlocks * 64];
        for (int i = 0; i < numBlocks; i++) {
            System.arraycopy(seed, 0, res, i*64, 64);
        }
        return res;
    }

//    public void inference_buffer_test() {
//        int[] range = {10, 31, 100, 316, 1000, 3162, 10000, 31622, 100000, 316227, 1000000, 3162277, 10000000, 31622776, 100000000};
//        for (int i = 0; i < range.length; i++) {
//            int numBlocks = range[i];
//            int[][] ciphertext = new int[numBlocks][16];
//            TensorBuffer ciphertextBuffer = TensorBuffer.createFixedSize(new int[]{numBlocks, 16},
//                    this.tflite.getOutputTensor(0).dataType());
//            long startTimeForReference = SystemClock.uptimeMillis();
//            int[][] plaintext = generate_input(numBlocks);
////            IntBuffer plaintextBuffer = IntBuffer.wrap(plaintext);
//            long endTimeForReference = SystemClock.uptimeMillis();
//            long input_generation_time = (endTimeForReference - startTimeForReference);
//        }
//    }
    public void inference_buffer_test0(){
        int[] plaintext = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                107, 193, 190, 226, 46, 64, 159, 150, 233, 61, 126, 17, 115, 147, 23, 42};
        IntBuffer ciphertextBuffer = IntBuffer.wrap(new int[plaintext.length]);
        IntBuffer plaintextBuffer = IntBuffer.wrap(plaintext);
        tflite.resizeInput(0, new int[]{2, 16});
        tflite.run(plaintextBuffer, ciphertextBuffer);
        Log.d(TAG, "ciphertext:" + Arrays.toString(ciphertextBuffer.array()));
    }

    public void generate_buffer_input_test() {
        Log.d(TAG, "block size:" + 2);
        Log.d(TAG, Arrays.toString(generate_buffer_input(2)));
    }

    public void inference_buffer_test() {
        int[] range = {10, 31, 100, 316, 1000, 3162, 10000, 31622, 100000, 316227, 1000000, 3162277, 10000000, 31622776, 100000000};
        for (int i = 0; i < range.length; i++) {

            int numBlocks = range[i];
            int[] ciphertext = new int[numBlocks * 16];
            IntBuffer ciphertextBuffer = IntBuffer.wrap(ciphertext);
            long startTimeForReference = SystemClock.uptimeMillis();
            int[] plaintext = generate_buffer_input(numBlocks);
            long endTimeForReference = SystemClock.uptimeMillis();
            long input_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer plaintextBuffer = IntBuffer.wrap(plaintext);
            endTimeForReference = SystemClock.uptimeMillis();
            long buffer_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            this.tflite.resizeInput(0, new int[]{numBlocks, 16});
            this.tflite.run(plaintextBuffer, ciphertextBuffer);
            endTimeForReference = SystemClock.uptimeMillis();
            long inference_time = (endTimeForReference - startTimeForReference);
            Log.d(TAG, numBlocks + ": " + input_generation_time + ", "  +buffer_generation_time + ", " + inference_time);
            System.out.println(getNumThreads() + "," + numBlocks + "," + input_generation_time + ","  +buffer_generation_time + "," + inference_time);
//            plaintext = null;
//            ciphertext = null;
//            System.gc();
        }
    }

    public void inference_buffer_test_sequential() {
        int[] range = {10, 31, 100, 316, 1000, 3162, 10000, 31622, 100000, 316227, 1000000, 3162277, 10000000, 31622776, 100000000};
        int[] length = new int[1];
        for (int i = 0; i < range.length; i++) {

            int numBlocks = range[i];
            length[0] = numBlocks;
            int[] ciphertext = new int[numBlocks * 16];
            IntBuffer ciphertextBuffer = IntBuffer.wrap(ciphertext);
            long startTimeForReference = SystemClock.uptimeMillis();
            int[] plaintext = generate_buffer_input(numBlocks);
            long endTimeForReference = SystemClock.uptimeMillis();
            long input_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer plaintextBuffer = IntBuffer.wrap(plaintext);
            IntBuffer lengthBuffer = IntBuffer.wrap(length);
            endTimeForReference = SystemClock.uptimeMillis();
            long buffer_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer[] inputs = {plaintextBuffer, lengthBuffer};
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, ciphertextBuffer);
            this.tflite.resizeInput(this.tflite.getInputIndex("plaintext"), new int[]{numBlocks, 16});
            this.tflite.runForMultipleInputsOutputs(inputs, outputs);
//            this.tflite.resizeInput(0, new int[]{numBlocks, 16});
//            IntBuffer[] inputs = {plaintextBuffer, lengthBuffer};
//            this.tflite.run(plaintextBuffer, ciphertextBuffer);
            endTimeForReference = SystemClock.uptimeMillis();
            long inference_time = (endTimeForReference - startTimeForReference);
            Log.d(TAG, numBlocks + ": " + input_generation_time + ", "  +buffer_generation_time + ", " + inference_time);
            System.out.println(getNumThreads() + "," + numBlocks + "," + input_generation_time + ","  +buffer_generation_time + "," + inference_time);
//            plaintext = null;
//            ciphertext = null;
//            System.gc();
        }
    }

    private int[] generate_counter(int[] counter, int numBlocks) {
//        int[] seed = {107, 193, 190, 226, 46, 64, 159, 150, 233, 61, 126, 17, 115, 147, 23, 42};
        int[] res = new int[numBlocks * 16];
        for (int i = 0; i < numBlocks; i++) {
            System.arraycopy(counter, 0, res, i*16, 16);
            for (int j = 15; j >= 0; j--) {
                counter[j]++;
                if (counter[j] < 256) {
                    break;
                } else {
                    counter[j] = 0;
                }
            }
        }
        return res;
    }

    private int[] chacha_generate_counter(int[] key, int[] nonce, int numBlocks) {
        int[] res = new int[numBlocks * 64];
//        int[] counter =  new int[64];
//        int[] tie = new int[] {101, 120, 112, 97, 110, 100, 32, 49, 54, 45, 98, 121, 116, 101, 32, 107};
//        for (int i = 0; i < numBlocks; i++) {
//
//        }
        int[] ctr = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < numBlocks; i++) {
            System.arraycopy(ctr, 0, res, i*16, 16);
        }
        return res;
    }

    public void inference_counter_outside() {
        int[] ctr;
        for (int i = 0; i < range.length; i++) {
            ctr = new int[]{240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255};
            int numBlocks = range[i];
            int[] ciphertext = new int[numBlocks * 16];
            IntBuffer ciphertextBuffer = IntBuffer.wrap(ciphertext);

            long startTimeForReference = SystemClock.uptimeMillis();
            int[] plaintext = generate_buffer_input(numBlocks);
            long endTimeForReference = SystemClock.uptimeMillis();
            long input_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            int[] counter = generate_counter(ctr, numBlocks);
            endTimeForReference = SystemClock.uptimeMillis();
            long counter_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer plaintextBuffer = IntBuffer.wrap(plaintext);
            IntBuffer ctrBuffer = IntBuffer.wrap(counter);
            endTimeForReference = SystemClock.uptimeMillis();
            long buffer_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer[] inputs = {ctrBuffer, plaintextBuffer};
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, ciphertextBuffer);
            this.tflite.resizeInput(this.tflite.getInputIndex("counter"), new int[]{numBlocks, 16});
            this.tflite.resizeInput(this.tflite.getInputIndex("text_input"), new int[]{numBlocks, 16});
            this.tflite.runForMultipleInputsOutputs(inputs, outputs);
            endTimeForReference = SystemClock.uptimeMillis();
            long inference_time = (endTimeForReference - startTimeForReference);
            Log.d(TAG, numBlocks + ": " + input_generation_time + "," + counter_generation_time  + ", "  +buffer_generation_time + ", " + inference_time);
            System.out.println(getNumThreads() + "," + numBlocks + "," + input_generation_time + "," + counter_generation_time + ","  + +buffer_generation_time + "," + inference_time);

        }
    }

    public void inference_counter_outside_sequential() {
        int[] ctr;
        int[] length = new int[1];
        for (int i = 0; i < range.length; i++) {
            ctr = new int[]{240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255};
            int numBlocks = range[i];
            length[0] = numBlocks;
            int[] ciphertext = new int[numBlocks * 16];
            IntBuffer ciphertextBuffer = IntBuffer.wrap(ciphertext);

            long startTimeForReference = SystemClock.uptimeMillis();
            int[] plaintext = generate_buffer_input(numBlocks);
            long endTimeForReference = SystemClock.uptimeMillis();
            long input_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            int[] counter = generate_counter(ctr, numBlocks);
            endTimeForReference = SystemClock.uptimeMillis();
            long counter_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer plaintextBuffer = IntBuffer.wrap(plaintext);
            IntBuffer ctrBuffer = IntBuffer.wrap(counter);
            IntBuffer lengthBuffer = IntBuffer.wrap(length);
            endTimeForReference = SystemClock.uptimeMillis();
            long buffer_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer[] inputs = {ctrBuffer, plaintextBuffer, lengthBuffer};
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, ciphertextBuffer);
            this.tflite.resizeInput(this.tflite.getInputIndex("counter"), new int[]{numBlocks, 16});
            this.tflite.resizeInput(this.tflite.getInputIndex("text_input"), new int[]{numBlocks, 16});
            this.tflite.runForMultipleInputsOutputs(inputs, outputs);
            endTimeForReference = SystemClock.uptimeMillis();
            long inference_time = (endTimeForReference - startTimeForReference);
            Log.d(TAG, numBlocks + ": " + input_generation_time + "," + counter_generation_time  + ", "  +buffer_generation_time + ", " + inference_time);
            System.out.println(getNumThreads() + "," + numBlocks + "," + input_generation_time + "," + counter_generation_time + ","  + +buffer_generation_time + "," + inference_time);

        }
    }

    public void inference_chacha20() {
        int[] key = new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        int[] nonce = new int[] {101, 102, 103, 104, 105, 106, 107, 108};
        int[] round_keys = new int[] {43, 126, 21, 22, 40, 174, 210, 166, 171, 247, 21, 136, 9, 207, 79, 60, 160, 250, 254, 23, 136, 84, 44, 177, 35, 163, 57, 57, 42, 108, 118, 5, 242, 194, 149, 242, 122, 150, 185, 67, 89, 53, 128, 122, 115, 89, 246, 127, 61, 128, 71, 125, 71, 22, 254, 62, 30, 35, 126, 68, 109, 122, 136, 59, 239, 68, 165, 65, 168, 82, 91, 127, 182, 113, 37, 59, 219, 11, 173, 0, 212, 209, 198, 248, 124, 131, 157, 135, 202, 242, 184, 188, 17, 249, 21, 188, 109, 136, 163, 122, 17, 11, 62, 253, 219, 249, 134, 65, 202, 0, 147, 253, 78, 84, 247, 14, 95, 95, 201, 243, 132, 166, 79, 178, 78, 166, 220, 79, 234, 210, 115, 33, 181, 141, 186, 210, 49, 43, 245, 96, 127, 141, 41, 47, 172, 119, 102, 243, 25, 250, 220, 33, 40, 209, 41, 65, 87, 92, 0, 110, 208, 20, 249, 168, 201, 238, 37, 137, 225, 63, 12, 200, 182, 99, 12, 166};
        for (int i = 0; i < range.length; i++) {
            int numBlocks = range[i];
            int[] ciphertext = new int[numBlocks * 64];
            IntBuffer ciphertextBuffer = IntBuffer.wrap(ciphertext);

            long startTimeForReference = SystemClock.uptimeMillis();
            int[] plaintext = chacha_generate_buffer_input(numBlocks);
            long endTimeForReference = SystemClock.uptimeMillis();
            long input_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            int[] keystream = chacha_generate_counter(key, nonce, numBlocks);
            endTimeForReference = SystemClock.uptimeMillis();
            long counter_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer plaintextBuffer = IntBuffer.wrap(plaintext);
            IntBuffer keysteamBuffer = IntBuffer.wrap(keystream);
            endTimeForReference = SystemClock.uptimeMillis();
            long buffer_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer[] inputs = {keysteamBuffer, plaintextBuffer};
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, ciphertextBuffer);
            this.tflite.resizeInput(this.tflite.getInputIndex("keystream"), new int[]{numBlocks, 64});
            this.tflite.resizeInput(this.tflite.getInputIndex("plaintext"), new int[]{numBlocks, 64});
            this.tflite.runForMultipleInputsOutputs(inputs, outputs);
//            this.tflite.runForMultipleInputsOutputs(inputs, outputs);
            endTimeForReference = SystemClock.uptimeMillis();
            long inference_time = (endTimeForReference - startTimeForReference);
            Log.d(TAG, numBlocks + ": " + input_generation_time + "," + counter_generation_time  + ", "  +buffer_generation_time + ", " + inference_time);
            System.out.println(getNumThreads() + "," + numBlocks + "," + input_generation_time + "," + counter_generation_time + ","  + +buffer_generation_time + "," + inference_time);

        }
    }

    public void inference_salsa20() {
        int[] key = new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        int[] nonce = new int[] {101, 102, 103, 104, 105, 106, 107, 108};
        int[] round_keys = new int[] {43, 126, 21, 22, 40, 174, 210, 166, 171, 247, 21, 136, 9, 207, 79, 60, 160, 250, 254, 23, 136, 84, 44, 177, 35, 163, 57, 57, 42, 108, 118, 5, 242, 194, 149, 242, 122, 150, 185, 67, 89, 53, 128, 122, 115, 89, 246, 127, 61, 128, 71, 125, 71, 22, 254, 62, 30, 35, 126, 68, 109, 122, 136, 59, 239, 68, 165, 65, 168, 82, 91, 127, 182, 113, 37, 59, 219, 11, 173, 0, 212, 209, 198, 248, 124, 131, 157, 135, 202, 242, 184, 188, 17, 249, 21, 188, 109, 136, 163, 122, 17, 11, 62, 253, 219, 249, 134, 65, 202, 0, 147, 253, 78, 84, 247, 14, 95, 95, 201, 243, 132, 166, 79, 178, 78, 166, 220, 79, 234, 210, 115, 33, 181, 141, 186, 210, 49, 43, 245, 96, 127, 141, 41, 47, 172, 119, 102, 243, 25, 250, 220, 33, 40, 209, 41, 65, 87, 92, 0, 110, 208, 20, 249, 168, 201, 238, 37, 137, 225, 63, 12, 200, 182, 99, 12, 166};
        for (int i = 0; i < range.length; i++) {
            int numBlocks = range[i];
            int[] ciphertext = new int[numBlocks * 64];
            IntBuffer ciphertextBuffer = IntBuffer.wrap(ciphertext);

            long startTimeForReference = SystemClock.uptimeMillis();
            int[] plaintext = chacha_generate_buffer_input(numBlocks);
            long endTimeForReference = SystemClock.uptimeMillis();
            long input_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            int[] keystream = chacha_generate_counter(key, nonce, numBlocks);
            endTimeForReference = SystemClock.uptimeMillis();
            long counter_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer plaintextBuffer = IntBuffer.wrap(plaintext);
            IntBuffer keysteamBuffer = IntBuffer.wrap(keystream);
            endTimeForReference = SystemClock.uptimeMillis();
            long buffer_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer[] inputs = {keysteamBuffer, plaintextBuffer};
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, ciphertextBuffer);
            this.tflite.resizeInput(this.tflite.getInputIndex("keystream"), new int[]{numBlocks, 64});
            this.tflite.resizeInput(this.tflite.getInputIndex("plaintext"), new int[]{numBlocks, 64});
            this.tflite.runForMultipleInputsOutputs(inputs, outputs);
//            this.tflite.runForMultipleInputsOutputs(inputs, outputs);
            endTimeForReference = SystemClock.uptimeMillis();
            long inference_time = (endTimeForReference - startTimeForReference);
            Log.d(TAG, numBlocks + ": " + input_generation_time + "," + counter_generation_time  + ", "  +buffer_generation_time + ", " + inference_time);
            System.out.println(getNumThreads() + "," + numBlocks + "," + input_generation_time + "," + counter_generation_time + ","  + +buffer_generation_time + "," + inference_time);

        }
    }

    public void inference_crypto_counter_outside() {
        int[] ctr;
        int[] round_keys = new int[] {43, 126, 21, 22, 40, 174, 210, 166, 171, 247, 21, 136, 9, 207, 79, 60, 160, 250, 254, 23, 136, 84, 44, 177, 35, 163, 57, 57, 42, 108, 118, 5, 242, 194, 149, 242, 122, 150, 185, 67, 89, 53, 128, 122, 115, 89, 246, 127, 61, 128, 71, 125, 71, 22, 254, 62, 30, 35, 126, 68, 109, 122, 136, 59, 239, 68, 165, 65, 168, 82, 91, 127, 182, 113, 37, 59, 219, 11, 173, 0, 212, 209, 198, 248, 124, 131, 157, 135, 202, 242, 184, 188, 17, 249, 21, 188, 109, 136, 163, 122, 17, 11, 62, 253, 219, 249, 134, 65, 202, 0, 147, 253, 78, 84, 247, 14, 95, 95, 201, 243, 132, 166, 79, 178, 78, 166, 220, 79, 234, 210, 115, 33, 181, 141, 186, 210, 49, 43, 245, 96, 127, 141, 41, 47, 172, 119, 102, 243, 25, 250, 220, 33, 40, 209, 41, 65, 87, 92, 0, 110, 208, 20, 249, 168, 201, 238, 37, 137, 225, 63, 12, 200, 182, 99, 12, 166};
        for (int i = 0; i < range.length; i++) {
            ctr = new int[]{240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255};
            int numBlocks = range[i];
            int[] ciphertext = new int[numBlocks * 16];
            IntBuffer ciphertextBuffer = IntBuffer.wrap(ciphertext);

            long startTimeForReference = SystemClock.uptimeMillis();
            int[] plaintext = generate_buffer_input(numBlocks);
            long endTimeForReference = SystemClock.uptimeMillis();
            long input_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            int[] counter = generate_counter(ctr, numBlocks);
            endTimeForReference = SystemClock.uptimeMillis();
            long counter_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer plaintextBuffer = IntBuffer.wrap(plaintext);
            IntBuffer ctrBuffer = IntBuffer.wrap(counter);
            IntBuffer roundKeyBuffer = IntBuffer.wrap(round_keys);
            endTimeForReference = SystemClock.uptimeMillis();
            long buffer_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer[] inputs = {ctrBuffer, plaintextBuffer, roundKeyBuffer};
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, ciphertextBuffer);
            this.tflite.resizeInput(this.tflite.getInputIndex("counter"), new int[]{numBlocks, 16});
            this.tflite.resizeInput(this.tflite.getInputIndex("plaintext"), new int[]{numBlocks, 16});
            this.tflite.resizeInput(this.tflite.getInputIndex("round_keys"), new int[]{176});
            this.tflite.runForMultipleInputsOutputs(inputs, outputs);
            endTimeForReference = SystemClock.uptimeMillis();
            long inference_time = (endTimeForReference - startTimeForReference);
            Log.d(TAG, numBlocks + ": " + input_generation_time + "," + counter_generation_time  + ", "  +buffer_generation_time + ", " + inference_time);
            System.out.println(getNumThreads() + "," + numBlocks + "," + input_generation_time + "," + counter_generation_time + ","  + +buffer_generation_time + "," + inference_time);

        }
    }

    public void inference_non_ecb_mode() {
//        this.tfliteModel = FileUtil.loadMappedFile(activity, );
        int[] iv = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

        int[] length = new int[1];
        for (int i = 0; i < range.length; i++) {
            int numBlocks = range[i];
            length[0] = numBlocks;
            int[] ciphertext = new int[numBlocks * 16];
            IntBuffer ciphertextBuffer = IntBuffer.wrap(ciphertext);

            long startTimeForReference = SystemClock.uptimeMillis();
            int[] plaintext = generate_buffer_input(numBlocks);
            long endTimeForReference = SystemClock.uptimeMillis();
            long input_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer plaintextBuffer = IntBuffer.wrap(plaintext);
            IntBuffer ivBuffer = IntBuffer.wrap(iv);
            IntBuffer lengthBuffer = IntBuffer.wrap(length);
            endTimeForReference = SystemClock.uptimeMillis();
            long buffer_generation_time = (endTimeForReference - startTimeForReference);

            startTimeForReference = SystemClock.uptimeMillis();
            IntBuffer[] inputs = {ivBuffer, plaintextBuffer, lengthBuffer};
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, ciphertextBuffer);
            this.tflite.resizeInput(this.tflite.getInputIndex("text_input"), new int[]{numBlocks, 16});
            this.tflite.runForMultipleInputsOutputs(inputs, outputs);
            endTimeForReference = SystemClock.uptimeMillis();
            long inference_time = (endTimeForReference - startTimeForReference);
            Log.d(TAG, numBlocks + ": " + input_generation_time + ", "  +buffer_generation_time + ", " + inference_time);
            System.out.println(getNumThreads() + "," + numBlocks + "," + input_generation_time + ","  +buffer_generation_time + "," + inference_time);
        }
    }


    public void input_generation_test(){
        int[] range = {10, 31, 100, 316, 1000, 3162, 10000, 31622, 100000, 316227, 1000000, 3162277, 10000000, 31622776, 100000000};
        for (int i = 0; i < range.length; i++) {
            int numBlocks = range[i];
            long startTimeForReference = SystemClock.uptimeMillis();
            generate_input(numBlocks);
            long endTimeForReference = SystemClock.uptimeMillis();
            Log.d(TAG, numBlocks + ": " + (endTimeForReference - startTimeForReference));
        }
    }


    public void test() {
        Log.d(TAG, "input shape:" + Arrays.toString(tflite.getInputTensor(0).shape()));
        Log.d(TAG, "output shape:" + Arrays.toString(tflite.getOutputTensor(0).shape()));
        int[][] plaintext, ciphertext;
        plaintext = new int[][]{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
        ciphertext = new int[2][16];
        tflite.run(plaintext, ciphertext);
        Log.i(TAG, "ciphertext:" + Arrays.deepToString(ciphertext));
        Log.i(TAG, "DataType:" + tflite.getInputTensor(0).dataType() +
                tflite.getOutputTensor(0).dataType());
    }

    private String getModelPath(String modelName) {
//        String filepath = "file://android_asset/ecb_cipher1.tflite";
        String filepath = null;
        if (modelName.equals("ecb")) {
            filepath = "ecb_cipher.tflite";
        } else if (modelName.equals("cbc")) {
            filepath = "cbc_cipher.tflite";
        } else if (modelName.equals("cfb")) {
            filepath = "cfb_cipher.tflite";
        } else if (modelName.equals("ofb")) {
            filepath = "ofb_cipher.tflite";
        } else if (modelName.equals("ctr")) {
            filepath = "ctr_cipher.tflite";
        } else if (modelName.equals("ctr_outside")) {
            filepath = "ctr_cipher_outside.tflite";
        } else if (modelName.equals("ecb_sequential")) {
            filepath = "ecb_cipher_sequential.tflite";
        } else if (modelName.equals("ctr_outside_sequential")) {
            filepath = "ctr_cipher_outside_sequential.tflite";
        } else if (modelName.equals("crypto_ctr_cipher_outside")) {
            filepath = "crypto_ctr_cipher_outside.tflite";
        } else if (modelName.equals("salsa20")) {
            filepath = "salsa20_cipher.tflite";
        } else if (modelName.equals("chacha20")) {
            filepath = "chacha20_cipher.tflite";
        }

//        File f = new File(filepath);
//        if (f.exists()) {
//            Log.i(TAG, "model file exists");
//        } else {
//            Log.e(TAG, "model file doesn't exist");
////            throw new IOException("Model file doesn't exist");
//        }
        return filepath;
    }

    /** Closes the interpreter and model to release resources. */
    public void close() {
        if (tflite != null) {
            // TODO: Close the interpreter
            tflite.close();
            tflite = null;
        }
        // TODO: Close the GPU delegate
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }

        tfliteModel = null;
    }

    public int getNumThreads() {
        return numThreads;
    }

    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
    }
}
