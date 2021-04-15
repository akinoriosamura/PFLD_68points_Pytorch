import cv2
import numpy as np
import onnx
from onnx import helper
from onnxsim import simplify
from onnx2keras import onnx_to_keras
import onnxruntime as rt
from pytorch2keras.converter import pytorch_to_keras
import tensorflow as tf
from tensorflow.python.keras import backend as K
import torch
from torch.autograd import Variable
from keras.models import Model

from model2 import MobileNetV2, BlazeLandMark, AuxiliaryNet, WingLoss, EfficientLM, HighResolutionNet, MyResNest50


def inf_onnx(img_path, image_size, onnx_model_path):
    img = np.array(cv2.imread(img_path))
    img = cv2.resize(img, (image_size, image_size))
    input_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = input_data.astype(np.float32) / 256.0
    input_data = np.expand_dims(input_data, 0)
    input_data = input_data.transpose((0, 3, 1, 2))

    session = rt.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    pred = session.run([output_name], {input_name: input_data})
    pred = pred[0][0].reshape(-1, 2) * [image_size, image_size]
    for (x, y) in pred.astype(np.int32):
        cv2.circle(img, (x, y), 1, (0, 0, 255), 2)
    cv2.imwrite('samples/labeled_onnx.jpg', img)

def get_onnx_output(input_data, onnx_model_path):
    input_data = input_data.transpose((0, 3, 1, 2))

    session = rt.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    # import pdb;pdb.set_trace()
    output_name = session.get_outputs()[-1].name
    pred = session.run([output_name], {input_name: input_data})[0]

    return pred

def compare_value(img_path, image_size, onnx_model, keras_model):
    # prepare input
    img = np.array(cv2.imread(img_path))
    img = cv2.resize(img, (image_size, image_size))
    input_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = input_data.astype(np.float32) / 256.0
    input_data = np.expand_dims(input_data, 0)

    # import pdb;pdb.set_trace()
    # get onnx model intermidiate layer value
    intermediate_tensor_name = "1458"
    model_path = 'models2/wflw_moruhard_grow_68_Res50/all_model_85_intermidiate.onnx'
    intermediate_layer_value_info = helper.ValueInfoProto()
    intermediate_layer_value_info.name = intermediate_tensor_name
    onnx_model.graph.output.append(intermediate_layer_value_info)
    onnx.save(onnx_model, model_path)

    onnx_pred = get_onnx_output(input_data, model_path)

    # get keras model intermidiate layer value
    layer_name = '1458'
    intermediate_layer_model = Model(inputs=keras_model.input,
                                    outputs=keras_model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(input_data)

    # check value
    # import pdb;pdb.set_trace()
    trans_onnx_pred = onnx_pred.transpose(0, 2, 3, 1)
    # onnx2kerasで2次元reshapeの前のtransposeのpermを[0, 3, 2, 1]→[0, 2, 3, 1]変更
    # fc前も2次元圧縮があるので、reshapelayerにtranspose入れて次元[0, 3, 1, 2]に
    

def main():
    # all model
    model = MyResNest50(nums_class=136)
    model = torch.load('models2/wflw_moruhard_grow_68_Res50/all_model_85.pth', map_location='cpu')

    # ONNX形式でのモデルの保存
    onnx_model_path = 'models2/wflw_moruhard_grow_68_Res50/all_model_85.onnx'
    dummy_input = torch.randn(1, 3, 128, 128)
    input_names = ['x']
    output_names = ['out']
    torch.onnx.export(model, dummy_input, onnx_model_path,
                    input_names=input_names, output_names=output_names)
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # convert simplify model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx_simp_model_path = 'models2/wflw_moruhard_grow_68_Res50/all_model_85_opt.onnx'
    onnx.save(model_simp, onnx_simp_model_path)

    onnx_simp_model = onnx.load(onnx_simp_model_path)
    onnx.checker.check_model(onnx_simp_model)

    input_names = ['x']
    # change_ordering=True で NCHW形式のモデルをNHWC形式のモデルに変換できる
    k_model = onnx_to_keras(onnx_model=onnx_simp_model, input_names=input_names,
                            change_ordering=True, verbose=True)

    ####### compare onnx value with keras
    # compare_value('samples/croped.jpg', 128, onnx_simp_model, k_model)

    k_model.save('models2/wflw_moruhard_grow_68_Res50/all_model_85_opt.h5')
    saved_model_dir = 'models2/wflw_moruhard_grow_68_Res50/all_model_85_opt_saved_model'
    tf.saved_model.save(k_model, saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    with open('models2/wflw_moruhard_grow_68_Res50/all_model_85_opt.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    main()