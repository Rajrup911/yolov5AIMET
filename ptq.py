import torch
from tqdm import tqdm
import numpy as np
import os

import ultralytics
from ultralytics.models.yolo.model import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data import build_dataloader, build_yolo_dataset

import cv2
from PIL import Image

from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.model_preparer import prepare_model
from aimet_torch.model_validator.model_validator import ModelValidator

class aimetPTQ:
    def __init__(self, config) -> None:
        self.config = config
        self.model = self.get_model() 
        self.dummy_input = torch.rand(1, 3, 640, 640).cuda()

    def get_model(self):
        model = YOLO(self.config.model.call_name, task = 'detect').cuda()
                                  
        #dummy_input = torch.rand(1, *self.config.model.input_shape).cuda()
        #torch.onnx.export(model, dummy_input, os.path.join(self.config.model.artifacts, "resnet_fp32.onnx"), export_params=True, opset_version=13, do_constant_folding=True)
        return model
    
    def validate(self, model):
        self.args = dict(model = model, data = '/home/ava/rajrup/yolov5/coco128.yaml')
        metrics = model.val(data='/home/ava/rajrup/yolov5/coco128.yaml')
                
    def prepare_model(self, model):
        return prepare_model(model)
    
    def cross_layer_equalization_auto(self, model):
        input_shape = (1, *self.config.model.input_shape)
        
        dummy_input = torch.rand(1, *self.config.model.input_shape).cuda()
        print(model.model)
        ModelValidator.validate_model(model.model, model_input=self.dummy_input)
        model = self.prepare_model(model.model)
        #equalize_model(model.cuda(), input_shape)
        #torch.onnx.export(model, dummy_input, os.path.join(self.config.model.artifacts, "resnet_after_CLE.onnx"), export_params=True, opset_version=13, do_constant_folding=True)
        #_ = fold_all_batch_norms(model, input_shapes = input_shape)
    
    def make_sim(self, model):
    
        use_cuda = False
        if torch.cuda.is_available():
            use_cuda = True
            
        sim = QuantizationSimModel(model.cuda(), quant_scheme = QuantScheme.post_training_tf_enhanced, default_output_bw=8, default_param_bw=8, 
                                   dummy_input = torch.rand(1, *self.config.model.input_shape).cuda())
                                   
        sim.set_and_freeze_param_encodings(encoding_path=os.path.join(self.config.model.artifacts, 'adaround.encodings'))
        sim.compute_encodings(forward_pass_callback = self.pass_calibration_data, forward_pass_callback_args = use_cuda)
        
        #dummy_input = torch.rand(1, *self.config.model.input_shape).cpu()
        #sim.export(path = self.config.model.artifacts, filename_prefix='resnet18_adaround', dummy_input=dummy_input)
        return sim
        
    def adaround(self, model):
        val_dataset = build_yolo_dataset(self.args, self.config.model.dataset_dir, 1)
        val_loader = build_dataloader(val_dataset, batch_size, 4, shuffle=False, rank=-1)
        
        params = AdaroundParameters(data_loader = val_loader, num_batches = 1, default_num_iterations = 1000)
    
        dummy_input = torch.rand(1, *self.config.model.input_shape).cuda()
        ada_model = Adaround.apply_adaround(model.cuda(), dummy_input, params,
                                            path = self.config.model.artifacts,
                                            filename_prefix = 'adaround',
                                            default_param_bw = 8,
                                            default_quant_scheme = QuantScheme.post_training_tf_enhanced)
        return ada_model
        
    def pass_calibration_data(self, sim_model, use_cuda):
    
        val_dataset = build_yolo_dataset(self.args, self.config.model.dataset_dir, 1)
        val_loader = build_dataloader(val_dataset, batch_size, 4, shuffle=False, rank=-1)
    
        batch_size = 1
        max_batch_counter = 50
    
        if use_cuda:
            device = torch.device('cuda')
    
        current_batch_counter = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
    
                inputs_batch = images.to(device)
                sim_model(inputs_batch)
    
                current_batch_counter += 1
                if current_batch_counter == max_batch_counter:
                    break