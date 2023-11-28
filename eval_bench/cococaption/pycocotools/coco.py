from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = 'tylin'
__version__ = '1.0.1'
from refile import smart_open

import json
import datetime

class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = []
        self.videoToAnns = {}
        self.videos = []
        if not annotation_file == None:
            print('loading annotations into memory...')
            time_t = datetime.datetime.utcnow()
            if 's3:/' in annotation_file:
                dataset = json.load(smart_open(annotation_file, 'r'))
            else:
                dataset = json.load(open(annotation_file, 'r'))
            if 'type' not in dataset:
                dataset['type']='caption'
            print(datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        videoToAnns = {ann['video_id']: [] for ann in self.dataset['annotations']}
        for ann in self.dataset['annotations']:
            videoToAnns[ann['video_id']] += [ann]
        videos = {vi['video_id']: {} for vi in self.dataset['annotations']}
        print('index created!')

        self.videoToAnns = videoToAnns
        self.videos = videos


    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in list(self.datset['info'].items()):
            print('%s: %s'%(key, value))

    def getVideoIds(self):
        return list(self.videos.keys())
        
    
    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()

        print('Loading and preparing results...     ')
        time_t = datetime.datetime.utcnow()
        if 's3:/' in resFile:
            anns = json.load(smart_open(resFile))
        else:
            anns = json.load(open(resFile))

        assert type(anns) == list, 'results in not an array of objects'
        annsVideoIds = [ann['video_id'] for ann in anns]
        assert set(annsVideoIds) == (set(annsVideoIds) & set(self.getVideoIds())), \
               'Results do not correspond to current coco set'

        print('DONE (t=%0.2fs)'%((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res
