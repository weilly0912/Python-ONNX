## ONNX
# WPI Confidential Proprietary
#--------------------------------------------------------------------------------------
# Copyright (c) 2020 Freescale Semiconductor
# Copyright 2020 WPI
# All Rights Reserved
##--------------------------------------------------------------------------------------
# * Code Ver : 1.0
# * Code Date: 2021/12/06
# * Author   : Weilly Li
#--------------------------------------------------------------------------------------
# THIS SOFTWARE IS PROVIDED BY WPI-TW "AS IS" AND ANY EXPRESSED OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL WPI OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#--------------------------------------------------------------------------------------

#(1) keras to onnx
# https://keras.io/
def Keras_to_ONNX( keras_h5_model ):
    import onnxmltools
    from keras.models import load_model
    input_keras_model = keras_h5_model
    output_onnx_model = keras_h5_model[:-2] + "onnx"
    keras_model = load_model(input_keras_model)
    onnx_model  = onnxmltools.convert_keras(keras_model) 
    onnxmltools.utils.save_model(onnx_model, output_onnx_model)
    print(output_onnx_model, 'is transfered done')
    return onnx_model

#(2) caffe to onnx
# https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/caffe_coreml_onnx.ipynb
# https://caffe.berkeleyvision.org/
def Caffe_to_ONNX( caffe_model, prototxt ):
    import onnx
    import coremltools
    import onnxmltools
    coreml_model = coremltools.converters.caffe.convert((caffe_model, prototxt)) 
    onnx_model  = onnxmltools.convert_coreml(coreml_model)
    output_onnx_model = caffe_model[:-10] + "onnx"
    onnxmltools.utils.save_model(onnx_model, output_onnx_model)
    print(output_onnx_model, 'is transfered done')
    return onnx_model

def Caffe_to_ONNX_instruction( prototxt, caffe_model, output_onnx_model ):
    #"!python -m caffe2onnx.convert --prototxt {prototxt} --caffemodel {caffe_model} --onnx {output_onnx_model}"
    str_out = "!python -m caffe2onnx.convert --prototxt " + prototxt + " --caffemodel " + caffe_model + " --onnx " + output_onnx_model
    return str_out

#(3) pythorch to onnx
# https://github.com/onnx/tutorials/blob/master/tutorials/PytorchOnnxExport.ipynb
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
def Pytorch_to_ONNX( torch_model, pytorch_model_size, output_onnx_model):
    import torch
    torch.onnx.export(torch_model, pytorch_model_size, output_onnx_model)
    print(output_onnx_model, 'is transfered done')

#(4) microsoft cntk to onnx
# https://docs.microsoft.com/en-us/cognitive-toolkit/
def CNTK_to_ONNX( CTNK_model ):
    import cntk 
    input_cntk_model = CTNK_model
    output_onnx_model = CTNK_model[:-5] + "onnx"
    cntk_model = cntk.Function.load(input_cntk_model, device=cntk.device.cpu()) #loaing
    cntk_model.save(output_onnx_model, format=cntk.ModelFormat.ONNX)
    print(output_onnx_model, 'is transfered done')
    return onnx_model

#(5) mxnet to onnx
# https://mxnet.apache.org/versions/1.8.0/
# https://github.com/onnx/tutorials/blob/master/tutorials/MXNetONNXExport.ipynb
def MXNet_to_ONNX( mxnet_json, mxnet_params, mxnet_model_size ):
    import mxnet as mx
    import numpy as np
    from mxnet.contrib import onnx as onnx_mxnet
    input_mxnet_symbol = mxnet_json
    input_mxnet_params = mxnet_params
    input_shape = (mxnet_model_size[0], mxnet_model_size[1], mxnet_model_size[2], mxnet_model_size[3])
    output_onnx_model = mxnet_json[:-4] + 'onnx'
    onnx_mxnet.export_model(input_mxnet_symbol, input_mxnet_params, [input_shape], np.float32, output_onnx_model)
    print(output_onnx_model, 'is transfered done')

#(6) scikit-learn
# https://scikit-learn.org/stable/
# http://onnx.ai/sklearn-onnx/index.html
def Sklearn_to_ONNX( sklearn_model , output_name ):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    onnx = convert_sklearn(clr, initial_types=[('float_input', FloatTensorType([None, 4]))])
    output_onnx_model = output_name
    with open(output_onnx_model, "wb") as f:
        f.write(onnx.SerializeToString())
    print(output_onnx_model, 'is transfered done')

#(7) coreml to onnx
# https://developer.apple.com/documentation/coreml
# https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/coreml_onnx.ipynb
def CoreML_to_ONNX( coreml_model ):
    import coremltools
    import onnxmltools
    input_coreml_model = coreml_model
    output_onnx_model = coreml_model[:-7] + 'onnx'
    coreml_model = coremltools.utils.load_spec(input_coreml_model)
    onnx_model = onnxmltools.convert_coreml(coreml_model)
    onnxmltools.utils.save_model(onnx_model, output_onnx_model)
    print(output_onnx_model, 'is transfered done')
    return onnx_model

#(8) tensorflow to onnx (ver 1.x)
# https://www.tensorflow.org/
# https://github.com/onnx/tensorflow-onnx
# https://github.com/onnx/tutorials/blob/master/tutorials/TensorflowToOnnx-1.ipynb
def TensorFlow_pb_to_ONNX( tensorflow_pb_model ):
    output_onnx_model = tensorflow_pb_model[:-2] + 'onnx'
    # "!python -m tf2onnx.convert --input {tensorflow_pb_model} --inputs x:0 --outputs prediction:0 --output {output_onnx_model} --verbose"
    str_out = "!python -m tf2onnx.convert --input " + tensorflow_pb_model + " --inputs x:0 --outputs prediction:0 --output " + output_onnx_model + " --verbose "
    #print(output_onnx_model, 'is transfered done')
    return str_out  
  

#(9) chainer to onnx 
# https://github.com/onnx/tutorials/blob/master/tutorials/ChainerOnnxExport.ipynb
def Chainer_to_ONNX( chainer_model, model_size, output_onnx_model ):
    import chainer
    import numpy as np
    import onnx_chainer
    x = np.zeros(model_size, dtype=np.float32)
    chainer.config.train = False # Don't forget to set train flag off!
    onnx_chainer.export( chainer_model, x, filename=output_onnx_model)
    print(output_onnx_model, 'is transfered done')

#(10) onnx to keras
# https://github.com/gmalivenko/onnx2keras
def ONNX_to_Keras( ONNX_Model, input_name ):
    import onnx
    import keras
    from onnx2keras import onnx_to_keras
    onnx_model = onnx.load(ONNX_Model)
    k_model = onnx_to_keras(onnx_model, [input_name])
    keras_model = ONNX_Model[:-4] + 'h5'
    keras.models.save_model(k_model, keras_model,overwrite=True,include_optimizer=True)
    return k_model
    print(output_onnx_model, 'is transfered done')

#(11) onnx to caffe2
# https://github.com/onnx/tutorials/blob/master/tutorials/OnnxCaffe2Import.ipynb
def ONNX_to_Caffe2( ONNX_Model, img ):
    import onnx
    import caffe2.python.onnx.backend
    model = onnx.load(ONNX_Model) # Load the ONNX model
    outputs = caffe2.python.onnx.backend.run_model(model, [img]) # Run the ONNX model with Caffe2
    return outputs

#(12) onnx to pytorch
# https://pythonrepo.com/repo/fumihwh-onnx-pytorch
def ONNX_to_Pytorch( ONNX_Model ):
    from onnx_pytorch import code_gen
    code_gen.gen(ONNX_Model, "./")

#(14) onnx to mxnet
#https://github.com/onnx/tutorials/blob/master/tutorials/OnnxMxnetImport.ipynb
def ONNX_to_MXNet( ONNX_Model, input_size, input_name, OutputPath  ):
    import mxnet as mx
    import mxnet.contrib.onnx as onnx_mxnet
    sym, arg_params, aux_params = onnx_mxnet.import_model(ONNX_Model)
    mod = mx.mod.Module(symbol=sym, data_names=[input_name], label_names=None)
    mod.bind(for_training=False, data_shapes=[(input_name , input_size)])
    mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True, allow_extra=True)
    mod.save_checkpoint(OutputPath,0)
    print(OutputPath, 'is transfered done')
    return mod

#(15) onnx to coreml
# https://github.com/onnx/tutorials/blob/master/tutorials/OnnxCoremlImport.ipynb
def ONNX_to_CoreML( ONNX_Model, input_name, output_name):
    import sys
    from onnx import onnx_pb
    from onnx_coreml import convert
    model_file = open(ONNX_Model, 'rb')
    model_proto = onnx_pb.ModelProto()
    model_proto.ParseFromString(model_file.read())
    coreml_model = convert(model_proto, image_input_names=[input_name], image_output_names=[output_name])
    coreml_model.save( ONNX_Model[:-4] + 'mlmodel' )
    print(ONNX_Model[:-4] + 'mlmodel', 'is transfered done')
    return coreml_model
  
#(16) onnx to tensorflow
# https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowImport.ipynb
def ONNX_to_TensorFlow( ONNX_Model ):
    import onnx
    from onnx_tf.backend import prepare
    onnx_model = onnx.load(ONNX_Model)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(ONNX_Model[:-4] + 'pb')
    print(ONNX_Model[:-4] + 'pb', 'is transfered done')
    return tf_rep