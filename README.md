﻿# **Python-ONNX**  
 [COLAB Source Code](https://colab.research.google.com/drive/1VPU9IMdRNcpd9rn2b8Ves1u_36inELnP?usp=sharing)
 
 (1) Setting & Installing Enviroment
 
     pip install -r /content/requirement.txt
 
 (2) Using Function 
 
     ** Example : Kera to ONNX**
     
     import onnx_model_transform
     
     onnx_model = onnx_model_transform.Keras_to_ONNX( "/content/drive/MyDrive/Colab_Model_ZOO/ONNX_Model_Zoo/CNN_Keras.h5" )
 
  -----------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://github.com/weilly0912/Python-ONNX/blob/ver1.0/ONNX_Model_Transform.png)
-----------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://github.com/weilly0912/Python-ONNX/blob/ver1.0/ONNX_Format.JPG)
-----------------------------------------------------------------------------------------------------------------------------------------------------------

**[Open Neural Network Exchange_ONNX.ipynb](https://github.com/weilly0912/Python-ONNX/blob/ver1.0/Open%20Neural%20Network%20Exchange_ONNX.ipynb)**

(1) ONNX Runtime 使用  
     - 如何執行 ONNX 模組   
     - 安裝相對應的版本  

(2) ONNX Runtime 實際範例 - 物件偵測  
     - RCNN ONNX Model  

(3) 將模組轉換 ONNX 格式  
     - keras to onnx  
     - caffe to onnx  
     - pyTorch to onnx  
     - microsoft cntk to onnx  
     - sklearn to onnx  
     - coreml to onnx  
     - tf to onnx  
     - tflite to onnx  
     - lightgbm to onnx  
     - lightgbm to onnx  

(4) 將 ONNX 格式轉換成其他模組格式  
     - onnx to tensorflow  
     - onnx to coreml  
     - onnx to caffe2  
     - onnx to keras  
 
(5) ONNX Using Load / Save / Optimiz  
     - 進行模組優化  

(6) ONNX i.MX8  
     - RCNN ONNX Model 實現物件偵測範例  

(7) 實際應用紀錄 - ONNX to Tensorflow Lite  
     - ONNX to Tensoflow  
     - Keras to Tensorflow Lite  
     
