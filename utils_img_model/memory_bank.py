
import random

class MemoryBank(object):
    def __init__(self, n_classes, source_list_file = None, buffer_size = None):
        self.n_classes = n_classes
        self.buffer_size = buffer_size  #  the capacity of the buffer
        self._parse_list(source_list_file)
        self._init_buffer()

    def _parse_list(self, list_file):
        self.sample_list = []  # todo a list of img_path and label
        if isinstance(list_file, str):
            for line in open(list_file):
                items = line.strip('\n').split(' ')
                img_path, label = items[0], int(items[1])
                # vidname, imgname = img_path.split('/')[-2], img_path.split('/')[-1].split('.')[0]
                # self.sample_list.append( (f'{vidname}_{imgname}', label) )
                self.sample_list.append( (img_path, label) )
        elif isinstance(list_file, list):
            self.sample_list = list_file


    def _init_buffer(self):
        # the buffer that collects all the features in the source domain
        self.buffer_dict = dict()
        for idx in range(self.n_classes):
            self.buffer_dict.update( {idx: dict()}) # a dictionary that contains all the classes

    def update_buffer(self, batch_index, batch_feat):
        # todo momentum update for the features in the buffer
        batch_feat = batch_feat.detach()
        # todo sample_unique_index is index in the source_data_list
        for sample_id, unique_index in enumerate(batch_index):
            # vidname, imgname = img_path.split('/')[-2], img_path.split('/')[-1].split('.')[0]
            label = self.sample_list[unique_index][1]  # sample_list is a list of tuple of (vidname_framename, vid_label )
            self.buffer_dict[label].update( { unique_index  : batch_feat[sample_id] } )  # todo how to update the features in the buffer, moving average or just re-write

    def random_sample(self, class_id):
        # randomly select a positive sample (from the same class) and a negative sample (from a different class )
        neg_sample_list = []
        for idx in range( self.n_classes):
            if idx == class_id:
                pos_sample_list = list(self.buffer_dict[idx].values() )
            else:
                neg_sample_list += list(self.buffer_dict[idx].values() )  # aggregate all the negative samples (in a different class )
        if len(pos_sample_list) != 0 and len(neg_sample_list) != 0:
            pos_sample, neg_sample = random.sample(pos_sample_list, 1)[0], random.sample(neg_sample_list,1)[0]
        else:
            pos_sample, neg_sample = None, None
        return pos_sample, neg_sample

# class MemoryBankForVideoFrame(object):
#     """
#     the memory bank for source video frames,  the list of source video frames is updated in every epoch
#     """
#     def __init__(self, n_classes, source_list_file = None, buffer_size = None):
#         self.n_classes = n_classes
#         self.buffer_size = buffer_size  #  the capacity of the buffer
#         self._parse_list(source_list_file)
#         self._init_buffer()