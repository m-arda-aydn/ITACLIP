import torch
import torch
import torch.nn as nn
import sys 
sys.path.append("..")
import cv2
from clip import clip
from prompts.imagenet_template import openai_imagenet_template
import torchvision.transforms as T
import numpy as np
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData

from mmseg.registry import MODELS
from pamr import PAMR


@MODELS.register_module()
class ITACLIP_Segmentor(BaseSegmentor):
    def __init__(self, model_name, name_path, dataset_name, device=torch.device('cuda'), pretrained = None,
                    train_cfg = None, pamr_steps=0, pamr_stride=(8, 16), prob_thd=0.0, logit_scale=40, 
                    slide_stride=112, slide_crop=224, area_thd=None, img_engineering = False, auxiliary_text_path = None, 
                    attn_self = True, def_coefficient=0.2, img_eng_coefficient=0.75, width_chunk_size = None):
        
        assert dataset_name in ['coco_stuff','coco_object','voc21','context60']
        bg = False
        if dataset_name in ['coco_object','voc21','context60']:
            bg = True # sets True when the dataset contains the "background" class.
        self.bg = bg
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            rgb_to_bgr=True)
        super().__init__(data_preprocessor=data_preprocessor)
        self.device = device
        self.net, _ = clip.load(model_name, device=self.device, jit=False)
        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)
        self.img_engineering = img_engineering
        self.transforms = ([T.Grayscale(),
                            T.GaussianBlur(kernel_size=11,sigma=5)]) # first-category augmentations
        self.flip_transforms = ([T.RandomVerticalFlip(p=1),
                                 T.RandomHorizontalFlip(p=1)]) # second-category augmentations
        self.attn_self = attn_self # self-self attention
        self.def_coefficient = def_coefficient
        self.img_eng_coefficient = img_eng_coefficient
        self.width_chunk_size = width_chunk_size # This variable is used to reduce GPU memory usage when num_cls != num_queries

        if auxiliary_text_path is None:
            self.query_features = self.text_feature(query_words)
        else:
            auxiliary_texts = self.get_aux_text(auxiliary_text_path)
            original_features = self.text_feature(query_words)
            aux_features = self.text_feature(auxiliary_texts)
            if self.bg:
                self.query_features = torch.zeros_like(original_features)
                num_bg_words = (self.query_idx == 0).sum().item()
                aux_features = aux_features[self.query_idx[num_bg_words:] - 1]
                self.query_features[num_bg_words:] = (1 - self.def_coefficient) * original_features[num_bg_words:] + (self.def_coefficient) * aux_features
                self.query_features[:num_bg_words] = original_features[:num_bg_words]
            else:
                aux_features = aux_features[self.query_idx]
                self.query_features = (1 - self.def_coefficient) * original_features + (self.def_coefficient) * aux_features

        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.area_thd = area_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

        if pamr_steps > 0:
            self.pamr = PAMR(pamr_steps, dilations=pamr_stride).to(device)
        else:
            self.pamr = None
        
    def perform_in_chunks(self, seg_logits, query_idx, num_cls, num_queries, width_chunk_size=200):
        device = seg_logits.device
        height, width = seg_logits.shape[-2:]
        seg_logits = seg_logits.unsqueeze(0)
        output = torch.zeros((num_cls, height, width), device=device)
        cls_index = nn.functional.one_hot(query_idx)
        cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)

        for i in range(0, width, width_chunk_size):
            chunk_end = min(i + width_chunk_size, width)
            output[:,:,i:chunk_end] = (seg_logits[:,:,:,i:chunk_end] * cls_index).max(1)[0]
        
        return output

    def get_aux_text(self, path):
        aux_text = []
        with open(path,'r') as f:
            aux_text = f.readlines()
        for i,name in enumerate(aux_text):
            name = name.replace('\n','')
            aux_text[i] = name.split('>=')[1]

        return aux_text
    
    def get_flipped_logits(self, flip_logits, transforms, size, w, h, out_dim):
        logit_list = []
        for i,flip_logit in enumerate(flip_logits):
            flip_logit = flip_logit.permute(0, 2, 1).reshape(-1, out_dim, w, h)
            logit = nn.functional.interpolate(flip_logit, size=size, mode='bilinear')
            logit = transforms[i](logit)
            logit_list.append(logit)
        logits = torch.mean(torch.stack(logit_list),dim=0)
        return logits

    def forward_feature(self, img, text_features, logit_size=None):
        if type(img) == list:
            img = img[0]
        
        img_list = []
        flip_list = []
        if not self.img_engineering:
            image_features = self.net.encode_image(img, return_all=True, attn_self=self.attn_self, device=self.device)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            img_list.append(image_features)
        else:
            torch.manual_seed(0)
            image_features = self.net.encode_image(img, return_all=True, attn_self=self.attn_self, device=self.device)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            img_list.append(image_features)
            for transform in self.transforms:
                new_img = transform(img.squeeze())
                new_img = new_img.unsqueeze(0)
                if new_img.shape[1] == 1:
                    new_img = new_img.expand(1,3,-1,-1)
                image_features = self.net.encode_image(new_img, return_all=True, attn_self=self.attn_self, device=self.device)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                img_list.append(image_features)
            
            for transform in self.flip_transforms:
                new_img = transform(img.squeeze())
                new_img = new_img.unsqueeze(0)
                if new_img.shape[1] == 1:
                    new_img = new_img.expand(1,3,-1,-1)
                flipped_image_features = self.net.encode_image(new_img, return_all=True, attn_self=self.attn_self, device=self.device)
                flipped_image_features /= flipped_image_features.norm(dim=-1, keepdim=True)
                flip_list.append(flipped_image_features)

            image_features = torch.mean(torch.stack(img_list), dim=0)

        image_features = image_features[:, 1:] 
        logits = image_features @ text_features.T
        if self.img_engineering:
            flip_logit_list = []
            for flip_img_features in flip_list:
                flip_img_features = flip_img_features[:, 1:]
                flip_logit_list.append(flip_img_features @ text_features.T)

        patch_size = self.net.visual.patch_size
        w, h = img[0].shape[-2] // patch_size, img[0].shape[-1] // patch_size
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)

        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
            if self.img_engineering:
                flip_logits = self.get_flipped_logits(flip_logit_list,self.flip_transforms,
                                                      size=img.shape[-2:], w = w, h = h, out_dim = out_dim)
                logits = (self.img_eng_coefficient) * logits + (1 - self.img_eng_coefficient) * flip_logits
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')
            if self.img_engineering:
                flip_logits = self.get_flipped_logits(flip_logit_list,self.flip_transforms,
                                                      size=logit_size, w = w, h = h, out_dim = out_dim)
                logits = (self.img_eng_coefficient) * logits + (1 - self.img_eng_coefficient) * flip_logits
        return logits

    def text_feature(self, query_words, templates=openai_imagenet_template):
        query_features = []
        with torch.no_grad():
            for qw in query_words:
                query = clip.tokenize([temp(qw) for temp in templates]).to(self.device)
                feature = self.net.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        
            return torch.cat(query_features, dim=0)
    
    def forward_slide(self, img, img_metas, text_features, query_idx, pamr=None, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = len(query_idx)
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img), device=self.device)
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img), device=self.device)
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.forward_feature(crop_img, text_features=text_features)
                preds += nn.functional.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        if pamr:
            img = nn.functional.interpolate(img, size=img_size, mode='bilinear')
            self.pamr = self.pamr.to(self.device)
            logits = self.pamr(img, logits.to(img.dtype)).to(img.dtype)

        return logits

    def predict(self, inputs, data_samples):
        self.net = self.net.to(self.device)
        inputs = inputs.to(self.device)
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        
        if type(inputs) == list:
            inputs = inputs[0].unsqueeze(0)

        if self.slide_crop > 0:
            query_idx = self.query_idx
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.query_features, query_idx, self.pamr, self.slide_stride, self.slide_crop)
        else:
            query_idx = self.query_idx    
            seg_logits = self.forward_feature(inputs, self.query_features, batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(seg_logits, data_samples, query_idx)
    
    def postprocess_result(self, seg_logits, data_samples, query_idx):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0) # n_queries * w * h

            num_cls, num_queries = max(query_idx) + 1, len(query_idx)
            if num_cls != num_queries:
                if self.width_chunk_size is None:
                    seg_logits = seg_logits.unsqueeze(0)
                    cls_index = nn.functional.one_hot(query_idx)
                    cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                    seg_logits = (seg_logits * cls_index).max(1)[0]
                else:
                    width_chunk_size = self.width_chunk_size
                    seg_logits = self.perform_in_chunks(seg_logits, query_idx, num_cls, num_queries, width_chunk_size=width_chunk_size)

            if self.area_thd is not None:
                # Force segmentations with area < self.area_thd to 0 (background)
                predictions = nn.functional.one_hot(seg_logits.argmax(0), num_cls).to(seg_logits.dtype)
                area_pred = predictions[:, :, 1:].sum((0, 1), keepdim=True) 
                area_pred = (area_pred > self.area_thd * area_pred.sum()).to(seg_logits.dtype)          
                seg_logits[1:] *= area_pred.transpose(0, -1)
            
            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            seg_pred = torch.from_numpy(cv2.morphologyEx(seg_pred.squeeze().cpu().numpy().astype(np.uint8), cv2.MORPH_CLOSE, kernel)).unsqueeze(0)
        
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': seg_pred})
            })

        return data_samples
    
    def _forward(data_samples):
        """
        """
    
    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """
    
    def extract_feat(self, inputs):
        """
        """
    
    def loss(self, inputs, data_samples):
        """
        """

def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices
