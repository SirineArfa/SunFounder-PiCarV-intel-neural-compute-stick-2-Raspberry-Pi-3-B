import tensorflow as tf
model = tf.keras.models.load_model("/content/lane_navigation.h5")
tf.saved_model.save(model,'model')
model.summary()
print(model.layers[0].name, model.layers[0].get_input_shape_at(0))
print(model.layers[-1].name, model.layers[-1].get_output_at(0).name)
import openvino
from openvino import inference_engine
#!pip install --upgrade pip
#!pip install openvino
from openvino.inference_engine import IECore, IENetwork
ie = IECore()
model_xml = '/content/lane_navigation_final .xml'
model_bin = '/content/lane_navigation_final .bin'
net =  ie.read_network(model=model_xml, weights=model_bin)
exec_net =  ie.load_network(network=net, device_name="MYRIAD"
assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
assert len(net.outputs) == 1, "Sample supports only single output topologies"
input_blob = next(iter(net.input_info.keys()))
out_blob = next(iter(net.outputs))
net.batch_size = len([0])
net.input_info[input_blob].input_data.shape
res = exec_net.infer(inputs={input_blob: X_test[0]})
res[out_blob].reshape(1,10)
