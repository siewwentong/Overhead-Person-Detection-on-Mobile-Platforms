import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
import os

pipeline = pipeline_pb2.TrainEvalPipelineConfig()
config_path = 'a.config'
with tf.gfile.GFile(config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline)

pipeline.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(os.getcwd(), "records/train.record")]
pipeline.train_input_reader.label_map_path = os.path.join(os.getcwd(), "labels.pbtxt")
pipeline.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(os.getcwd(), "records/val.record")]
pipeline.eval_input_reader[0].label_map_path = os.path.join(os.getcwd(), "labels.pbtxt")
pipeline.train_config.fine_tune_checkpoint = os.path.join(os.getcwd(), "pretrained_mobilenet/v2_quant_300x300/model.ckpt")
pipeline.train_config.batch_size = 32
# CONFIGURE THE FOLLOWING
pipeline.train_config.num_steps = 50000
pipeline.model.ssd.num_classes = 2
# END OF CONFIGURE


# Enable ssdlite, this should already be enabled in the config we downloaded, but this is just to make sure.
pipeline.model.ssd.box_predictor.convolutional_box_predictor.kernel_size = 3
pipeline.model.ssd.box_predictor.convolutional_box_predictor.use_depthwise = True
pipeline.model.ssd.feature_extractor.use_depthwise = True
# Quantization Aware Training
pipeline.graph_rewriter.quantization.delay = 0
pipeline.graph_rewriter.quantization.weight_bits = 8
pipeline.graph_rewriter.quantization.activation_bits = 8

config_text = text_format.MessageToString(pipeline)
with tf.gfile.Open(config_path, "wb") as f:
    f.write(config_text)
