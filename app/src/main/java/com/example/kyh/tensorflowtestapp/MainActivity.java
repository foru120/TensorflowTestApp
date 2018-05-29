package com.example.kyh.tensorflowtestapp;

import android.content.ClipData;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int PICTURE_REQUEST_CODE = 100;

    // Tensorflow parameter
    private static final String[] RIGHT_INPUT_NAMES = {"right/input_scope/low_res_X","right/input_scope/mid_res_X","right/input_scope/high_res_X"};
    private static final String[] LEFT_INPUT_NAMES = {"left/input_scope/low_res_X","left/input_scope/mid_res_X","left/input_scope/high_res_X"};
    private static final String[] OUTPUT_NAMES = {"right_1/softmax","left_1/softmax"};
    private static final int[] WIDTHS = {160, 200, 240};
    private static final int[] HEIGHTS = {60, 80, 100};
    private static final String MODEL_FILE = "file:///android_asset/model_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/label_strings.txt";
    private static final int MULTISCALE_CNT = 3;

    private ImageView rightImg;
    private ImageView leftImg;
    private ImageView rightLowImg;
    private ImageView leftLowImg;
    private ImageView rightMidImg;
    private ImageView leftMidImg;
    private ImageView rightHighImg;
    private ImageView leftHighImg;

    private Button galleryBtn;
    private Button verificationBtn;
    private Button grayscaleBtn;
    private TextView resultText;

    private float[] lowRightData = new float[WIDTHS[0] * HEIGHTS[0]];
    private float[] midRightData = new float[WIDTHS[1] * HEIGHTS[1]];
    private float[] highRightData = new float[WIDTHS[2] * HEIGHTS[2]];
    private float[] lowLeftData = new float[WIDTHS[0] * HEIGHTS[0]];
    private float[] midLeftData = new float[WIDTHS[1] * HEIGHTS[1]];
    private float[] highLeftData = new float[WIDTHS[2] * HEIGHTS[2]];
//    private int[] testData = new int[WIDTHS[0] * HEIGHTS[0]];
    Bitmap bmpGrayScale = Bitmap.createBitmap(WIDTHS[0], HEIGHTS[0], Bitmap.Config.ARGB_4444);

    private Executor executor = Executors.newSingleThreadExecutor();

    private TensorFlowClassifier classifier = TensorFlowClassifier.getInstance();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        rightImg = (ImageView) findViewById(R.id.imgRight);
        leftImg = (ImageView) findViewById(R.id.imgLeft);
        rightLowImg = (ImageView) findViewById(R.id.imgLowRight);
        leftLowImg = (ImageView) findViewById(R.id.imgLowLeft);
        rightMidImg = (ImageView) findViewById(R.id.imgMidRight);
        leftMidImg = (ImageView) findViewById(R.id.imgMidLeft);
        rightHighImg = (ImageView) findViewById(R.id.imgHighRight);
        leftHighImg = (ImageView) findViewById(R.id.imgHighLeft);

        galleryBtn = (Button) findViewById(R.id.btnGallery);
        galleryBtn.setOnClickListener(new ButtonEventHandler());
        verificationBtn = (Button) findViewById(R.id.btnVerification);
        verificationBtn.setOnClickListener(new ButtonEventHandler());
        grayscaleBtn = (Button) findViewById(R.id.btnGrayscale);
        grayscaleBtn.setOnClickListener(new ButtonEventHandler());

        resultText = (TextView) findViewById(R.id.textResult);

        initTensorFlowAndLoadModel();
    }

    /**
     * 텐서플로우 classifier 초기화 함수
     */
    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try{
                    classifier.createClassifier(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            WIDTHS,
                            HEIGHTS,
                            RIGHT_INPUT_NAMES,
                            LEFT_INPUT_NAMES,
                            OUTPUT_NAMES);
                    Log.d(TAG, "Load Success");
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    /**
     * 버튼 이벤트를 처리하는 함수
     */
    public class ButtonEventHandler implements View.OnClickListener {
        @Override
        public void onClick(View v) {
            int objectID = v.getId();
            if (objectID == R.id.btnGallery) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Audio.Media.EXTERNAL_CONTENT_URI);
                intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
                intent.setType("image/*");
                startActivityForResult(Intent.createChooser(intent, "Select Image"), PICTURE_REQUEST_CODE);
            } else if (objectID == R.id.btnVerification) {
                int result = classifier.verificationEye(lowRightData, midRightData, highRightData, lowLeftData, midLeftData, highLeftData);
                resultText.setText("Class is : " + result);
            } else if (objectID == R.id.btnGrayscale) {
                rightImg.setImageBitmap(bmpGrayScale);
//                rightImg.setImageBitmap(BitmapFactory.decodeByteArray(testData, 0, WIDTHS[0] * HEIGHTS[0]));
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICTURE_REQUEST_CODE) {
            if (resultCode == RESULT_OK) {
                // 기존 이미지 삭제
                rightImg.setImageResource(0);
                leftImg.setImageResource(0);

                rightLowImg.setImageResource(0);
                leftLowImg.setImageResource(0);
                rightMidImg.setImageResource(0);
                leftMidImg.setImageResource(0);
                rightHighImg.setImageResource(0);
                leftHighImg.setImageResource(0);

                Uri uri = data.getData();
                ClipData clipData = data.getClipData();

                if (clipData != null) {  // "".equals(object): 문자열 null 값 비교
                    rightImg.setImageURI(clipData.getItemAt(0).getUri());
                    leftImg.setImageURI(clipData.getItemAt(1).getUri());
                    Bitmap oriRightBitmap = ((BitmapDrawable) rightImg.getDrawable()).getBitmap();
                    Bitmap oriLeftBitmap = ((BitmapDrawable) leftImg.getDrawable()).getBitmap();
                    Log.d(TAG, "right: (" + oriRightBitmap.getWidth() + "," + oriRightBitmap.getHeight() + "), left: (" + oriLeftBitmap.getWidth() + "," + oriLeftBitmap.getHeight() + ")");

                    for (int i = 0; i < MULTISCALE_CNT; i++) {
                        Bitmap tmpRightBitmap = Bitmap.createScaledBitmap(oriRightBitmap, WIDTHS[i], HEIGHTS[i], false);
                        Bitmap tmpLeftBitmap = Bitmap.createScaledBitmap(oriLeftBitmap, WIDTHS[i], HEIGHTS[i], false);
                        Log.d(TAG, i + "right: (" + tmpRightBitmap.getWidth() + "," + tmpRightBitmap.getHeight() + "), left: (" + tmpLeftBitmap.getWidth() + "," + tmpLeftBitmap.getHeight() + ")");
                        if (i == 0) {
                            lowRightData = grayScaleAndNorm(tmpRightBitmap);
                            lowLeftData = grayScaleAndNorm(tmpLeftBitmap);
                            rightLowImg.setImageBitmap(tmpRightBitmap);
                            leftLowImg.setImageBitmap(tmpLeftBitmap);
                        } else if (i == 1) {
                            midRightData = grayScaleAndNorm(tmpRightBitmap);
                            midLeftData = grayScaleAndNorm(tmpLeftBitmap);
                            rightMidImg.setImageBitmap(tmpRightBitmap);
                            leftMidImg.setImageBitmap(tmpLeftBitmap);
                        } else {
                            highRightData = grayScaleAndNorm(tmpRightBitmap);
                            highLeftData = grayScaleAndNorm(tmpLeftBitmap);
                            rightHighImg.setImageBitmap(tmpRightBitmap);
                            leftHighImg.setImageBitmap(tmpLeftBitmap);
                        }
                    }
                } else if (uri != null) {
                    rightImg.setImageURI(uri);
                }
            }
        }
    }

    private float[] grayScaleAndNorm(Bitmap bitmap) {
        int mWidth = bitmap.getWidth();
        int mHeight = bitmap.getHeight();

        int[] ori_pixels = new int[mWidth * mHeight];
        float[] norm_pixels = new float[mWidth * mHeight];

        bitmap.getPixels(ori_pixels, 0, mWidth, 0, 0, mWidth, mHeight);
        for (int i = 0; i < ori_pixels.length; i++) {
            int alpha = Color.alpha(ori_pixels[i]);
//            int grayPixel = (int) ((Color.red(ori_pixels[i]) * 0.2126) + (Color.green(ori_pixels[i]) * 0.7152) + (Color.blue(ori_pixels[i]) * 0.0722));
            int grayPixel = (int) ((Color.red(ori_pixels[i]) * 0.299) + (Color.green(ori_pixels[i]) * 0.587) + (Color.blue(ori_pixels[i]) * 0.114));  // decode_png -> grayscale 변환과 일치
            if (grayPixel < 0) grayPixel = 0;
            if (grayPixel > 255) grayPixel = 255;
            norm_pixels[i] = grayPixel / 255.0f;
        }
        return norm_pixels;
    }
}
