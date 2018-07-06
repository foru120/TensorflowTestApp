package com.example.kyh.tensorflowtestapp;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.support.v4.os.TraceCompat;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Vector;

public class TensorFlowClassifier {
    private static final String TAG = "TFClassifier";
    private static final int MAX_RESULTS = 3;  // result 개수 제한
    private static final float THRESHOLD = 0.1f;  // outputs 값의 threshold 설정

    // neural network 관련 parameters
    private String[] rightInputNames;  // neural network right 입력 노드 이름
    private String[] leftInputNames;  // neural network left 입력 노드 이름
    private String[] outputNames;  // neural network 출력 노드 이름
    private int[] widths;  // 입력 이미지 가로 길이
    private int[] heights;  // 입력 이미지 세로 길이
    private Vector<String> labels = new Vector<>();  // label 정보
    private int numClasses = 7;
    private float[][] logits = new float[2][numClasses];  // logit 정보
    private boolean runStats = false;

    private TensorFlowInferenceInterface tii;

    private TensorFlowClassifier() {

    }

    private static class SingleToneHolder {
        static final TensorFlowClassifier instance = new TensorFlowClassifier();
    }

    public static TensorFlowClassifier getInstance() {
        return SingleToneHolder.instance;
    }

    /**
     * 텐서플로우 classifier 생성 관련 초기화 함수
     * @param assetManager
     * @param modelFilename
     * @param labelFilename
     * @param widhts
     * @param heights
     * @param rightInputNames
     * @param leftInputNames
     * @param outputNames
     */
    public void createClassifier(
            AssetManager assetManager,
            String modelFilename,
            String labelFilename,
            int[] widhts,
            int[] heights,
            String[] rightInputNames,
            String[] leftInputNames,
            String[] outputNames) {
        this.rightInputNames = rightInputNames;
        this.leftInputNames = leftInputNames;
        this.outputNames = outputNames;
        this.widths = widhts;
        this.heights = heights;

        // label names 설정
        BufferedReader br = null;
        try {
            String actualFilename = labelFilename.split("file:///android_asset/")[1];
            br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
            String line = "";
            while((line = br.readLine()) != null) {
                this.labels.add(line);
            }
        } catch (IOException e) {
            Log.d(TensorFlowClassifier.TAG, e.toString());
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.d(TensorFlowClassifier.TAG, e.toString());
            }
        }

        this.tii = new TensorFlowInferenceInterface(assetManager, modelFilename);
    }

    /**
     * 오른쪽/왼쪽 눈에 대한 verification 을 수행하는 함수
     * @param lowRightData
     * @param midRightData
     * @param highRightData
     * @param lowLeftData
     * @param midLeftData
     * @param highLeftData
     * @return
     */
    public int verificationEye(float[] lowRightData, float[] midRightData, float[] highRightData,
                               float[] lowLeftData, float[] midLeftData, float[] highLeftData) {
        long startTime = System.currentTimeMillis();

        TraceCompat.beginSection("verificationEye");

        TraceCompat.beginSection("feed");
        for (int i = 0; i < this.widths.length; i++) {
            if (i == 0) {
                tii.feed(this.rightInputNames[i], lowRightData, 1, this.heights[i], this.widths[i], 1);
                tii.feed(this.leftInputNames[i], lowLeftData, 1, this.heights[i], this.widths[i], 1);
            } else if (i == 1) {
                tii.feed(this.rightInputNames[i], midRightData, 1, this.heights[i], this.widths[i], 1);
                tii.feed(this.leftInputNames[i], midLeftData, 1, this.heights[i], this.widths[i], 1);
            } else {
                tii.feed(this.rightInputNames[i], highRightData, 1, this.heights[i], this.widths[i], 1);
                tii.feed(this.leftInputNames[i], highLeftData, 1, this.heights[i], this.widths[i], 1);
            }
        }
//        tii.feed("right/input_scope/is_training", new boolean[]{false}, 1);
//        tii.feed("left/input_scope/is_training", new boolean[]{false}, 1);
        TraceCompat.endSection();

        TraceCompat.beginSection("run");
        tii.run(this.outputNames, this.runStats);
        TraceCompat.endSection();

        TraceCompat.beginSection("fetch");
        for (int i = 0; i < this.outputNames.length; i++) {
            float[] outputs = new float[this.numClasses];
            tii.fetch(this.outputNames[i], outputs);
            for (int j = 0; j < this.numClasses; j++) {
                this.logits[i][j] = outputs[j];
            }
        }
        TraceCompat.endSection();

        float[] result = {0f, 0f, 0f, 0f, 0f, 0f, 0f};
        for (int j = 0; j < this.logits[0].length; j++) {
            for (int i = 0; i < this.logits.length; i++) {
                result[j] += this.logits[i][j];
                Log.d(TAG, this.logits[i][j]+"");
            }
        }

        String temp = "";
        for (int i = 0; i < result.length; i++) {
            temp += result[i] + ",";
        }
        Log.d(TAG, temp);

        float maxValue = -1;
        int maxIndex = -1;
        for (int i = 0; i < this.numClasses; i++) {
            if (maxValue < result[i]) {
                maxValue = result[i];
                maxIndex = i;
            }
        }
        long endTime = System.currentTimeMillis();

        return maxIndex;
    }
}
