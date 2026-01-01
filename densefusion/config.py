import os
import torch
import numpy as np

class Config:
    def __init__(self):
        # Paths - update as needed
        self.LINEMOD_ROOT = "/content/datasets/linemod/Linemod_preprocessed_yolo_2"
        self.DATA_YAML_PATH = os.path.join(self.LINEMOD_ROOT, 'data.yaml')
        self.PLY_MODELS_DIR = "/content/datasets/linemod/Linemod_preprocessed_yolo_2/pose_models/models"
        self.DIAMETER_INFO_PATH = "/content/datasets/linemod/Linemod_preprocessed_yolo_2/pose_models/models_info.yml"
        self.MODELS_SAVE_DIR = "/content/models"
        self.CHECKPOINTS_DIR = os.path.join(self.LINEMOD_ROOT, 'checkpoints')

        # Device
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # Scales
        self.MODEL_SCALE_MM_TO_M = 0.001
        self.DEPTH_SCALE_MM_TO_M = 1000.0

        # Camera intrinsics (Linemod standard)
        self.K = np.array([
            [572.4114, 0,        325.2611],
            [0,        573.57043, 242.04899],
            [0,        0,        1        ]
        ], dtype=np.float32)

        # Objects
        self.OBJECT_IDS = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15']
        self.OBJECT_NAMES = ['ape', 'benchvise', 'camera', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']
        self.OBJECTS_TO_SKIP = ["03", "07"]

        # Dataset split
        self.TRAIN_RATIO = 0.7
        self.VAL_RATIO = 0.1
        self.TEST_RATIO = 0.2
        self.RANDOM_SEED = 42

        # Depth
        self.INCLUDE_DEPTH = True
        self.DEPTH_SUBFOLDER = "depth"

        # Symmetric objects
        self.SYMMETRIC_LIST = [7,8]

        # Model/training defaults
        self.USE_SEGMENTATION = True
        self.USE_TRANSFORMER_FUSION = False
        self.TRANSFORMER_HEADS = 2
        self.TRANSFORMER_LAYERS = 4
        self.TRANSFORMER_DIM = 128
        self.TRANSFORMER_DROPOUT = 0.1

        self.NUM_POINTS = 500
        self.PATCH_SIZE = 512

        self.BATCH_SIZE = 12
        self.NUM_EPOCHS = 15
        self.LEARNING_RATE = 1e-4
        self.USE_MIXED_PRECISION = True
        self.GRADIENT_ACCUMULATION_STEPS = 2

    def setup_environment(self):
        import torch.multiprocessing as mp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        try:
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    def verify_paths(self):
        paths = {
            'LINEMOD dataset': self.LINEMOD_ROOT,
            'PLY models': self.PLY_MODELS_DIR,
            'Diameter info': self.DIAMETER_INFO_PATH
        }
        all_good = True
        for name, path in paths.items():
            if os.path.exists(path):
                print(f"✓ {name}: {path}")
            else:
                print(f"✗ {name} NOT FOUND: {path}")
                all_good = False
        return all_good

    def print_config(self):
        print("="*60)
        print("DENSEFUSION CONFIGURATION")
        print("="*60)
        print(f"Device: {self.DEVICE}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Batch size: {self.BATCH_SIZE}")
        print(f"Epochs: {self.NUM_EPOCHS}")

# Singleton config
config = Config()
