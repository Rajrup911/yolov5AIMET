from benedict import benedict
import argparse
import json
import torch

from src.ptq import aimetPTQ

class JsonHandler:
    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            return benedict(data)

def main(config: benedict):

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        
    worker = aimetPTQ(config)    
    model = worker.get_model().cuda()
    
    worker.validate(model)
    
    worker.cross_layer_equalization_auto(model)
    model = worker.adaround(model)
    sim = worker.make_sim(model)
    
    worker.validate(sim.model)
    
if __name__ == "__main__":
    def get_argparse():
        parser = argparse.ArgumentParser(description='Create')
        parser.add_argument('--config', required=True,  type=str,   help = "")
        args = parser.parse_args()
        return args

    args = get_argparse()
    cfg = benedict(JsonHandler.load_json(args.config))
    cfg.args = vars(args)
    main(cfg)
