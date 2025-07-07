import sys
sys.path.append('..')
from utils.singletons.coco import COCO

if __name__ == '__main__':
    coco = COCO()
    coco.setup_dataset()