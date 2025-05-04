#!/usr/bin/env python3
import json
import numpy as np
from typing import Dict, Any

class COCODataHandler:
    """COCO-style image dataset handler

    The class handles COCO-style image dataset annotations and processing, 
    including image-caption embedding using Hugging Face's CLIP model and 
    multi-class one-hot label encoding.

    Attributes:
        _captions ()
    """
    def __init__(self, annotation: Dict[str, Any], preprocessed: bool=False):
        """Initializes instance of the class from image annotations.

        Args:
            annotations (dict): COCO-style annotations from json file.
            preprocessed (bool): To-be implemented later.
        """
        self._captions = {}
        self._categories = {}
        self._urls = {}

        self._annotations = annotation

        for data in annotation:
            coco_id = data['cocoId']
            coco_split = data['cocoSplit']
            url = f'http://images.cocodataset.org/{coco_split}/{coco_id.zfill(12)}.jpg'

            categories = set()
            captions = set()

            for category in data['categories']:
                categories.add(category['supercategory_name'])
            
            for caption in data['captions']:
                captions.add(caption)
            
            self._captions[coco_id] = captions
            self._categories[coco_id] = categories
            self._urls[coco_id] = url

        self._categories_one_hot, self._category_index = self.onehotencode(list(self._categories.values()))
    
    @property
    def captions(self) -> dict:
        """Returns a dictionary mapping COCO image IDs to their corresponding captions.

        Returns:
            dict: A dictionary mapping COCO image IDs to their corresponding captions.
        """
        return self._captions
    
    @property
    def categories(self) -> dict:
        """Returns a dictionary mapping COCO image IDs to their corresponding categories.
        
        Returns:
            dict: A dictionary mapping COCO image IDs to their corresponding categories.
        """
        return self._categories
    
    @property
    def urls(self) -> dict:
        """Returns a dictionary mapping COCO image IDs to their corresponding urls.
        
        Returns:
            dict: A dictionary mapping COCO image IDs to their corresponding image url.
        """
        return self._urls
    
    @property
    def categories_one_hot(self) -> np.array:
        """Returns the one-hot encoding for the image categories.
        
        Returns:
            np.array: A 2D number array where the rows represent the one hot encoding. 
        """
        return self._categories_one_hot
    
    @property
    def category_index(self) -> dict:
        """Returns the category indices used in one hot encoding.
        
        Returns:
            dict: A dictionary mapping the category to the index in one hot encoding. 
        """
        return self._category_index
    
    def __getitem__(self, coco_id: int) -> tuple:
        """Access the URL, caption and categories for the specific COCO ID.
        
        Args:
            coco_id (int): The COCO id of the associated image.
        Returns:
            tuple: A tuple containing the URL, captions and categories of the image.
        """
        return self._urls[coco_id], self._captions[coco_id], self._categories[coco_id]
    
    def __call__(self, *args, **kwds) -> tuple:
        """Access the URL, caption and categories for the specific index.
        
        Args:
            index (int): The index of the associated image.
        Returns:
            turple: A tuple containing the URL, captions and categories of the image.
        """
        index = args[0]
        values = self._annotations[index]

        coco_id = values['cocoId']
        coco_split = values['cocoSplit']

        categories = set()
        captions = set()
        url = f'http://images.cocodataset.org/{coco_split}/{coco_id.zfill(12)}.jpg'

        for category in values['categories']:
            categories.add(category['supercategory_name'])
        
        for caption in values['captions']:
            captions.add(caption)

        return url, captions, categories
    
    @classmethod
    def from_file(cls, file_path: str) -> 'COCODataHandler':
        """Create an instance from annotations file path.

        Args:
            file_path (str): The path to JSON file containing annotations.
        Returns:
            COCODataHandler: A new instance of COCODataHandler initialized using the annotations.
        """
        with open(file_path, 'r') as file:
            annotations = json.load(file)
        return cls(annotations)
    
    @staticmethod
    def onehotencode(categories: list) -> tuple:
        """One-hot encodes a list of categories.
        
        Args:
            categories (list): List containing categories for each image.
        Returns:
            tuple: A tuple containing: 
                - a numpy array of one hot encoded vectors 
                - adictionary mapping categories to indices.
        """
        unique_categories = set()
        for category in categories:
            unique_categories.update(category)
        unique_categories = sorted(unique_categories)

        category_index = {category: index for index, category in enumerate(unique_categories)}
        one_hot_encoded = []

        for img_category in categories:
            vector = np.zeros(len(unique_categories), dtype=int)

            for category in img_category:
                vector[category_index[category]] = 1
            one_hot_encoded.append(vector)
        
        return np.array(one_hot_encoded), category_index
    
    @staticmethod
    def decode_one_hot(one_hot: np.array, category_index:dict, threshold: float = 0.7) -> list:
        """Decodes the one-hot encoding back into category names above a given threshold.
        
        Args:
            one_hot (np.array): The one-hot encoded vector for a given image.
            threshold (float): The threshold above which a category is considered true.
        Returns:
            list: A list of corresponding category names.
        """
        categories = []

        for idx, vector in enumerate(one_hot):
            img_categories = { category for i, category in enumerate(category_index.keys()) 
                                if vector[i] >= threshold }
            categories.append(img_categories)
        
        return categories


if __name__ == '__main__':
    FAIL = '\033[91m'
    PASS = '\033[92m'
    ENDC = '\033[0m'

    # Test from_file
    object = COCODataHandler.from_file('./data/all-joined-1/COCO-Train17.json')
    print(f'Initializing Class From File:\t{PASS}Passed{ENDC}')

    # Test One Hot Decoding
    result = list(object.categories.values()) == COCODataHandler.decode_one_hot(object.categories_one_hot, object.category_index)
    if not result:
        print(f'One Hot Decoding:\t\t{FAIL}Failed{ENDC}')
        exit(1)
    print(f'One Hot Decoding:\t\t{PASS}Passed{ENDC}')

    # Test Indexing
    first_coco_id = list(object.captions.keys())[0]
    result = object(0) == object[first_coco_id]

    for idx, coco_id in enumerate(object.captions.keys()):
        result = object(idx) == object[coco_id]
        if not result:
            print(f'Indexing:\t\t\t{FAIL}Failed{ENDC} on index {idx}')
            exit(1)
    
    print(f'Indexing:\t\t\t{PASS}Passed{ENDC}')

    