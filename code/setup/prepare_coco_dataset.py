import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.singletons.coco import COCO

if __name__ == '__main__':
    coco = COCO()
    coco.setup_dataset()