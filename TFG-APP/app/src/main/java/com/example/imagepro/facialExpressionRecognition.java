package com.example.imagepro;


import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class facialExpressionRecognition {

    //Definir interprete
    //Pero antes de hacerlo hay que tener implementeado tensorflow para su funcionamiendo en build.gradle
    private Interpreter interpreter;
    private int INPUT_SIZE;
    //Definir la altura y anchura del frame original
    private int height=0;
    private int width=0;
    //Definir GpuDelegate para implementar la gpu en el interpreter
    private GpuDelegate gpuDelegate=null;
    //Por último definir cascadeClassifier para la detección de cara
    private CascadeClassifier cascadeClassifier;
    //Esta función es llamada en CameraActivity
    facialExpressionRecognition(AssetManager assetManager, Context context, String modelPath, int inputSize) throws IOException {
        INPUT_SIZE=inputSize;
        //Se establece el GPU para el interpreter
        Interpreter.Options options=new Interpreter.Options();
        gpuDelegate=new GpuDelegate();
        //Se añade gpuDelegate a opciones
        options.addDelegate(gpuDelegate);
        //Y ahora se establece el numero de threads a opciones
        options.setNumThreads(6);// El numero usado es de acuerdo a los telefonos que lo utilicen
        //Se añade el modelo al interpreter
        interpreter=new Interpreter(loadModelFile(assetManager,modelPath), options);
        //Si el modelo se carga
        Log.d("facial_Expression", "Model is loaded");
        //Ahora se carga el haarcascade classifier
        try {
            //Se define el inputStream para leer el classifier
            InputStream is = context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            //Se crea una carpeta
            File cascadeDir=context.getDir("cascade",Context.MODE_PRIVATE);
            //Y se añade un nuevo archivo en la carpeta
            File mCascadeFile=new File(cascadeDir, "haarcascade_frontalface_alt");
            //Se define el output Stream para transferir los datos al fichero creado
            FileOutputStream os= new FileOutputStream(mCascadeFile);
            //Ahora se crea un buffer para guardar byte
            byte[] buffer=new byte[4096];
            int byteRead;
            //Se lee el byte recibido en un while
            //Cuando lee -1 significa que no hay datos para leer
            while ((byteRead=is.read(buffer)) !=-1){
                //Se escribe el byte recibido a el fichero cascade
                os.write(buffer,0,byteRead);
            }
            //Se cierran los input y output
            is.close();
            os.close();

            cascadeClassifier=new CascadeClassifier(mCascadeFile.getAbsolutePath());
            //Si el fichero cascade se carga, se imprime
            Log.d("facial_Expression", "Classifier is loaded");


        }catch (IOException e){
            e.printStackTrace();
        }
    }
    //El input y output estan en formato Mat
    //Esta función se llama en onCameraframe de CameraActivity
    public Mat recognizeImage(Mat mat_image){
        //Antes de predecir la imagen no esta apropiadamente alineada
        //Hay que rotarla 90 grados para una buena predicción
        Core.flip(mat_image.t(),mat_image,1);//Rotar imagen 90 grados
        //Se empieza con el proceso y convierte la imagen a escala de grises
        Mat grayscaleImage=new Mat();
        Imgproc.cvtColor(mat_image, grayscaleImage, Imgproc.COLOR_RGBA2GRAY);
        //Se establece la altura y el ancho de la misma
        height= grayscaleImage.height();
        width= grayscaleImage.width();
        //Se define la altura minima de la cara en la imagen original
        int absoluteFaceSize=(int)(height*0.1);
        //Ahora se crea MatofRect para guardar la cara
        MatOfRect faces=new MatOfRect();
        //Primero hay que observar que el cascadeClassifier esté cargado
        if(cascadeClassifier !=null){
            //Detecta la cara en el frame
            //                                  input         output
            cascadeClassifier.detectMultiScale(grayscaleImage,faces,1.1,2,2,
                    new Size(absoluteFaceSize,absoluteFaceSize), new Size());
            //          tamaño mínimo
        }
        //Ahora se convierte en un array
        Rect[] faceArray= faces.toArray();
        //Se hace un loop para cada cara
        for (int i=0;i<faceArray.length;i++){
            //Se dibuja un rectangulo alrededor de la cara para su observación
            //                input/output-punto de inicio-punto final          color  R  G  B alpha    grosor
            Imgproc.rectangle(mat_image,faceArray[i].tl(),faceArray[i].br(),new Scalar(0,0,0,1),2);
            //Se recorta la cara del frame original y de la imagen en escala de grises
            //                 coordenada x                 coordenada y
            Rect roi=new Rect((int)faceArray[i].tl().x,(int)faceArray[i].tl().y,
                    ((int)faceArray[i].br().x)-(int)(faceArray[i].tl().x),
                    ((int)faceArray[i].br().y)-(int)(faceArray[i].tl().y));
            //Es importante mirarlo una vez mas
            Mat cropped_rgba=new Mat(mat_image,roi);
            //Ahora se convierte a bitmap
            Bitmap bitmap=null;
            bitmap=Bitmap.createBitmap(cropped_rgba.cols(),cropped_rgba.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(cropped_rgba,bitmap);
            //Se cambia el tamaño a (48,48)
            Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,48,48,false);
            //Y se convierte el bitmap escalado a bytebuffer
            ByteBuffer byteBuffer=convertBitmapToByteBuffer(scaledBitmap);
            //Se crea un objeto para mantener el output
            float[][] emotion=new float[1][1];
            //Ahora se predice con bytebuffer como entrada una emocion en salida
            interpreter.run(byteBuffer,emotion);
            //Define el valor float de la emoción
            float emotion_v=(float)Array.get(Array.get(emotion,0),0);
            //Si la emocion es reconocida se imprime el valor
            Log.d("facial_expression","Output: "+ emotion_v);
            //Se crea una funcion que devuelve el texto de la emoción
            String emotion_s=get_emotion_text(emotion_v);
            //Y se pone el texto de la emocion en el frame original(mat image)
            //              input/output   texto        valor de predicción
            Imgproc.putText(mat_image,"Emocion: "+emotion_s+") ",
                    new Point((int)faceArray[i].tl().x+10,(int)faceArray[i].tl().y+20),
                    1,2,new Scalar(255,255,255,1),2);
            //      escalar texto         color      R G  B  alpha   grosor
        }

        //Despues de la predicción se rota la imagen -90 grados
        Core.flip(mat_image.t(),mat_image,0);
        return mat_image;
    }

    private String get_emotion_text(float emotion_v) {
        //Se crea un string vacio
        String val="";
        //Se usa un if para determinar el valor
        if(emotion_v>=0 & emotion_v<0.2) {
            val = "Sorpresa";
        }else if(emotion_v>=0.2 & emotion_v < 1.4){
            val="Miedo";
        }else if(emotion_v>=1.4 & emotion_v < 2.6){
            val="Enfadado";
        }else if(emotion_v>=2.6 & emotion_v < 3.1){
            val="Neutral";
        }else if(emotion_v>=3.1 & emotion_v < 4.5){
            val="Triste";
        }else if(emotion_v>=4.5 & emotion_v < 5.1) {
            val = "Disgustado";
        }else{
            val="Feliz";
        }
        return val;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap scaledBitmap) {
        ByteBuffer byteBuffer;
        int size_image=INPUT_SIZE;

        byteBuffer=ByteBuffer.allocateDirect(4*1*size_image*size_image*3);
        //4 es multiplicado por el float de entrada
        //3 es multiplicado por rgb
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_image*size_image];
        scaledBitmap.getPixels(intValues,0,scaledBitmap.getWidth(),
                0,0,scaledBitmap.getWidth(),scaledBitmap.getHeight());

        int pixel=0;
        for(int i=0; i < size_image; i++){
            for (int j=0; j < size_image; j++){
                final int val=intValues[pixel++];
                //Ahora se pone el valor float a bytebuffer
                //Se escala la imagen para convertirla de 0-255 a 0-1
                byteBuffer.putFloat((((val>>16)&0xFF))/255.0f);
                byteBuffer.putFloat((((val>>8)&0xFF))/255.0f);
                byteBuffer.putFloat(((val&0xFF))/255.0f);

            }
        }
        return byteBuffer;
    }
    //Función para cargar el modelo
    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException{
        //Se añade descripcion al archivo
        AssetFileDescriptor assetFileDescriptor=assetManager.openFd(modelPath);
        //Se crea un inputStream para leer el archivo
        FileInputStream inputStream=new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();

        long startOffset=assetFileDescriptor.getStartOffset();
        long declaredLength=assetFileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);

    }

}
