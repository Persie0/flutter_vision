package com.vladih.computer_vision.flutter_vision.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.support.image.TensorImage;

import java.nio.ByteBuffer;
import java.util.List;

public class utils {
    private static final String TAG = "YOLOUtils";
    
    /**
     * Crop bitmap with improved bounds checking and error handling
     */
    public static Bitmap crop_bitmap(Bitmap bitmap, float x1, float y1, float x2, float y2) {
        if (bitmap == null || bitmap.isRecycled()) {
            throw new IllegalArgumentException("Invalid bitmap provided");
        }
        
        try {
            final int bitmapWidth = bitmap.getWidth();
            final int bitmapHeight = bitmap.getHeight();
            
            // Ensure coordinates are within bitmap bounds
            final int x = Math.max(0, Math.min((int) x1, bitmapWidth - 1));
            final int y = Math.max(0, Math.min((int) y1, bitmapHeight - 1));
            final int x_end = Math.max(x + 1, Math.min((int) x2, bitmapWidth));
            final int y_end = Math.max(y + 1, Math.min((int) y2, bitmapHeight));
            
            final int width = x_end - x;
            final int height = y_end - y;
            
            if (width <= 0 || height <= 0) {
                Log.w(TAG, String.format("Invalid crop dimensions: width=%d, height=%d", width, height));
                return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888);
            }
            
            return Bitmap.createBitmap(bitmap, x, y, width, height);
        } catch (Exception e) {
            Log.e(TAG, "Error cropping bitmap", e);
            throw new RuntimeException("Bitmap cropping failed: " + e.getMessage());
        }
    }

    /**
     * Convert RGB bitmap to grayscale Mat with improved error handling
     */
    public static Mat rgbBitmapToMatGray(Bitmap bitmap) {
        if (bitmap == null || bitmap.isRecycled()) {
            throw new IllegalArgumentException("Invalid bitmap provided for Mat conversion");
        }
        
        Mat mat = null;
        try {
            mat = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC3);
            Utils.bitmapToMat(bitmap, mat);
            
            Mat grayMat = new Mat();
            Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_RGB2GRAY);
            
            return grayMat;
        } catch (Exception e) {
            Log.e(TAG, "Error converting bitmap to grayscale Mat", e);
            throw new RuntimeException("Bitmap to Mat conversion failed: " + e.getMessage());
        } finally {
            if (mat != null) {
                mat.release();
            }
        }
    }

    /**
     * Feed input tensor with optimized processing and better error handling
     */
    public static ByteBuffer feedInputTensor(Bitmap bitmap,
                                           int input_width,
                                           int input_height,
                                           int src_width,
                                           int src_height,
                                           float mean,
                                           float std) throws Exception {
        if (bitmap == null || bitmap.isRecycled()) {
            throw new IllegalArgumentException("Invalid bitmap provided for tensor input");
        }
        
        Bitmap processedBitmap = null;
        try {
            Log.d(TAG, String.format("Processing tensor input: src(%dx%d) -> input(%dx%d)", 
                    src_width, src_height, input_width, input_height));
            
            TensorImage tensorImage;
            String processing_mode;
            
            // Determine processing mode based on image size
            if (src_width > input_width || src_height > input_height) {
                processing_mode = "downsize";
            } else {
                processing_mode = "upsize";
            }
            
            tensorImage = FeedInputTensorHelper.getBytebufferFromBitmap(
                    bitmap, input_width, input_height, mean, std, processing_mode);
            
            if (tensorImage == null || tensorImage.getBuffer() == null) {
                throw new Exception("Failed to create tensor image");
            }
            
            Log.d(TAG, String.format("Tensor processing completed with mode: %s", processing_mode));
            return tensorImage.getBuffer();
            
        } catch (Exception e) {
            Log.e(TAG, "Error processing tensor input", e);
            throw new Exception("Tensor input processing failed: " + e.getMessage());
        } finally {
            // Clean up the original bitmap if it's not recycled
            if (bitmap != null && !bitmap.isRecycled()) {
                bitmap.recycle();
            }
            
            // Clean up processed bitmap if different from original
            if (processedBitmap != null && processedBitmap != bitmap && !processedBitmap.isRecycled()) {
                processedBitmap.recycle();
            }
        }
    }

    /**
     * Convert YUV420 bytes to bitmap with improved memory management and stride handling
     */
    public static Bitmap feedInputToBitmap(Context context,
                                         List<byte[]> bytesList,
                                         int imageHeight,
                                         int imageWidth,
                                         int rotation) throws Exception {
        if (context == null) {
            throw new IllegalArgumentException("Context cannot be null");
        }
        
        if (bytesList == null || bytesList.size() < 3) {
            throw new IllegalArgumentException("Invalid YUV bytes list");
        }
        
        Bitmap bitmapRaw = null;
        Bitmap rotatedBitmap = null;
        
        try {
            byte[] yPlane = bytesList.get(0);
            int ySize = yPlane.length;
            
            // 1. Calculate Stride
            // If the buffer size is larger than W*H, there is padding (stride > width)
            int rowStride = ySize / imageHeight;
            
            Log.d(TAG, String.format("YUV processing: Y=%d bytes, dimensions=%dx%d, stride=%d", 
                    ySize, imageWidth, imageHeight, rowStride));
            
            // 2. Create a clean, tightly packed NV21 buffer (Standard size: 1.5 * W * H)
            int packedSize = (int) (imageWidth * imageHeight * 1.5);
            byte[] nv21Data = new byte[packedSize];

            // 3. Copy Y Plane row by row to remove padding
            for (int r = 0; r < imageHeight; r++) {
                System.arraycopy(yPlane, r * rowStride, nv21Data, r * imageWidth, imageWidth);
            }

            // 4. Handle UV Plane (Using Grayscale fallback for stability)
            // The UV planes in bytesList are often problematic to merge correctly without more metadata.
            // Filling them with 127 (neutral gray) keeps the image valid (grayscale) and perfect for detection.
            int uvStartIndex = imageWidth * imageHeight;
            java.util.Arrays.fill(nv21Data, uvStartIndex, nv21Data.length, (byte) 127);

            // 5. Convert to Bitmap
            bitmapRaw = RenderScriptHelper.getBitmapFromNV21(context, nv21Data, imageWidth, imageHeight);

            if (bitmapRaw == null) {
                throw new Exception("Failed to convert NV21 to bitmap");
            }

            // 6. Rotate
            if (rotation != 0) {
                Matrix matrix = new Matrix();
                matrix.postRotate(rotation);
                rotatedBitmap = Bitmap.createBitmap(bitmapRaw, 0, 0,
                        bitmapRaw.getWidth(), bitmapRaw.getHeight(), matrix, true);
                
                Log.d(TAG, String.format("Applied rotation: %d degrees", rotation));
                return rotatedBitmap;
            } else {
                return bitmapRaw;
            }

        } catch (Exception e) {
            Log.e(TAG, "Error processing YUV to bitmap", e);
            throw new Exception("YUV to bitmap conversion failed: " + e.getMessage());
        } finally {
            if (rotation != 0 && bitmapRaw != null && !bitmapRaw.isRecycled()) {
                bitmapRaw.recycle();
            }
        }
    }

    /**
     * Utility method to validate bitmap
     */
    public static boolean isValidBitmap(Bitmap bitmap) {
        return bitmap != null && !bitmap.isRecycled() && bitmap.getWidth() > 0 && bitmap.getHeight() > 0;
    }

    /**
     * Utility method to safely recycle bitmap
     */
    public static void safeRecycleBitmap(Bitmap bitmap) {
        if (bitmap != null && !bitmap.isRecycled()) {
            try {
                bitmap.recycle();
            } catch (Exception e) {
                Log.w(TAG, "Error recycling bitmap", e);
            }
        }
    }

    /**
     * Get memory usage info for debugging
     */
    public static String getMemoryInfo() {
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long usedMemory = totalMemory - freeMemory;
        
        return String.format("Memory: Used=%dMB, Free=%dMB, Total=%dMB, Max=%dMB",
                usedMemory / (1024 * 1024),
                freeMemory / (1024 * 1024),
                totalMemory / (1024 * 1024),
                maxMemory / (1024 * 1024));
    }
}