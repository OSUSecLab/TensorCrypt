package com.example.wb_aes;

import android.os.Bundle;

import com.example.wb_aes.tflite.Classifier;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;

import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;
import android.view.View;

import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.example.wb_aes.databinding.ActivityMainBinding;

import android.view.Menu;
import android.view.MenuItem;
import com.example.wb_aes.tflite.Classifier.Device;
import com.example.wb_aes.tflite.Classifier;

import java.io.IOException;
import java.nio.IntBuffer;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.Key;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;

public class MainActivity extends AppCompatActivity {

    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;
    private final String TAG = "Main";
    private Classifier classifier;
    private Device device = Device.CPU;
    private int numThreads = 4;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.d(TAG, "on create");
        super.onCreate(savedInstanceState);
//        for (int i = 1; i < 10; i++) {
//            setNumThreads(i);
//            createClassifier(getDevice(), getNumThreads());
////            this.classifier.inference_array_test();
//            this.classifier.inference_buffer_test();
//        }
//        Classifier classifier = new Classifier()

//        this.classifier.resizeInput(10);
//        this.classifier.input_generation_test();
//        this.classifier.inference_array_test();
//        this.classifier.test();
//        runInference("ctr_outside");
        runInference("chacha20");
//        runInference("ctr_outside_sequential");
//        runInference("crypto_ctr_cipher_outside");
//        runInference("ofb");
//        try {
//            javaCipherBaseline();
//        } catch (InvalidAlgorithmParameterException | InvalidKeyException | BadPaddingException | NoSuchAlgorithmException | IllegalBlockSizeException | NoSuchPaddingException e) {
//            e.printStackTrace();
//        }
//        this.classifier.generate_buffer_input_test();
    }

    private void runInference(String modelName) {
        createClassifier(getDevice(), getNumThreads(), modelName);
        Trace.beginSection("runInference");
        if (modelName.equals("ecb")) {
            this.classifier.inference_buffer_test();
        } else if (modelName.equals("ctr_outside")) {
            this.classifier.inference_counter_outside();
        } else if (modelName.equals("ecb_sequential")) {
            this.classifier.inference_buffer_test_sequential();
        } else if (modelName.equals("ctr_outside_sequential")) {
            this.classifier.inference_counter_outside_sequential();
        } else if (modelName.equals("crypto_ctr_cipher_outside")) {
            this.classifier.inference_crypto_counter_outside();
        } else if (modelName.equals("salsa20")) {
            this.classifier.inference_salsa20();
        } else if (modelName.equals("chacha20")) {
            this.classifier.inference_chacha20();
        }
        else {
            this.classifier.inference_non_ecb_mode();
        }

        Trace.endSection();
    }

    private void createClassifier(Device device, int numThreads, String modelName) {
        if (this.classifier != null) {
            Log.d(TAG,"Closing classifier.");
            this.classifier.close();
            this.classifier = null;
        }
        try {
            this.classifier = new Classifier(this, device, numThreads, modelName);
            Log.d(TAG, String.format("Creating classifier (device=%s, numThreads=%d)",device, numThreads));
        } catch (IOException e) {
            Log.e(TAG, "Failed to create classifier.");
        }
    }

    private byte[] generate_buffer_input(int numBlocks) {
        int[] source = {107, 193, 190, 226, 46, 64, 159, 150, 233, 61, 126, 17, 115, 147, 23, 42};
        byte[] seed = new byte[16];
        for (int i = 0; i < 16; i++) {
            seed[i] = (byte) source[i];
        }
        byte[] res = new byte[numBlocks * 16];
        for (int i = 0; i < numBlocks; i++) {
            System.arraycopy(seed, 0, res, i*16, 16);
        }
        return res;
    }

    private byte[] getBytesFromInts(int[] input) {
        byte[] seed = new byte[input.length];
        for (int i = 0; i < 16; i++) {
            seed[i] = (byte) input[i];
        }
        return seed;
    }

    public void javaCipherBaseline() throws NoSuchPaddingException, NoSuchAlgorithmException, InvalidAlgorithmParameterException, InvalidKeyException, IllegalBlockSizeException, BadPaddingException {
        int[] iv = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        int[] range = {10, 31, 100, 316, 1000, 3162, 10000, 31622, 100000, 316227, 1000000};
        int[] key = {43, 126, 21, 22, 40, 174, 210, 166, 171, 247, 21, 136, 9, 207, 79, 60};
//        int[] length = new int[1];

        // initialize cipher
        long startTimeForReference = SystemClock.uptimeMillis();
//        try {
//            SecureRandom secureRandom = new SecureRandom();
            Cipher cipher = Cipher.getInstance("AES/CTR/NoPadding");
            byte[] byteKey = getBytesFromInts(key);
            byte[] nonce = getBytesFromInts(iv);
            Key keySpec = new SecretKeySpec(byteKey, "AES");
            IvParameterSpec ivSpec = new IvParameterSpec(nonce);
            cipher.init(Cipher.ENCRYPT_MODE, keySpec, ivSpec);
//        } catch (Exception ignored) {
//            System.out.println("Ciper initialization failure!");
//        }

        long endTimeForReference = SystemClock.uptimeMillis();
        long buffer_generation_time = (endTimeForReference - startTimeForReference);

        for (int i = 0; i < range.length; i++) {
            int numBlocks = range[i];
//            length[0] = numBlocks;
//            int[] ciphertext = new int[numBlocks * 16];

//            IntBuffer ciphertextBuffer = IntBuffer.wrap(ciphertext);

            startTimeForReference = SystemClock.uptimeMillis();
            byte[] plaintext = generate_buffer_input(numBlocks);
            endTimeForReference = SystemClock.uptimeMillis();
            long input_generation_time = (endTimeForReference - startTimeForReference);



            startTimeForReference = SystemClock.uptimeMillis();
            byte[] ciphertext = cipher.doFinal(plaintext);
            endTimeForReference = SystemClock.uptimeMillis();
            long inference_time = (endTimeForReference - startTimeForReference);
            Log.d("baseline", numBlocks + ": " + input_generation_time + ", "  +buffer_generation_time + ", " + inference_time);
            System.out.println(getNumThreads() + "," + numBlocks + "," + input_generation_time + ","  +buffer_generation_time + "," + inference_time);
        }
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public boolean onSupportNavigateUp() {
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main);
        return NavigationUI.navigateUp(navController, appBarConfiguration)
                || super.onSupportNavigateUp();
    }

    public Device getDevice() {
        return device;
    }

    public void setDevice(Device device) {
//        this.device = device;
        if (this.device != device) {
            Log.d(TAG, "Updating  device: " + device);
            this.device = device;
        }
    }

    public int getNumThreads() {
        return numThreads;
    }

    public void setNumThreads(int numThreads) {
        if (this.numThreads != numThreads) {
            Log.d(TAG, "Updating  numThreads: " + numThreads);
            this.numThreads = numThreads;
        }
    }
}