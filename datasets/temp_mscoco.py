import os
import json
from PIL import Image
from torchvision.datasets import VisionDataset
import numpy as np

############ replace every words
# confall_tmp = '../datasets/mscoco/one_cap/coco_test_confall_tmp.txt'
# nothall_tmp = '../datasets/mscoco/one_cap/coco_test_nothall_tmp.txt'
confall = '../datasets/mscoco/one_cap/coco_test_confall.txt'
nothall = '../datasets/mscoco/one_cap/coco_test_nothall.txt'
corr = '../datasets/mscoco/mix/conf1_corr2345.txt'
confall_corr2345 = '../datasets/mscoco/mix/confall_corr2345.txt'
nothall_corr2345 = '../datasets/mscoco/mix/nothall_corr2345.txt'
imgs=[]
falls=[]
nalls=[]
corrs2=[]
corrs3=[]
corrs4=[]
corrs5=[]
with open(nothall) as f:
    for line in f:
        img, nall = line.split('++')
        imgs.append(img)
        nalls.append(nall.split('\n')[0])
    f.close()
with open(corr) as f:
    for line in f:
        caps = line.split('++')
        corrs2.append(caps[2])
        corrs3.append(caps[3])
        corrs4.append(caps[4])
        corrs5.append(caps[5])
    f.close()

with open(nothall_corr2345, "a") as fw:
    for i in range(len(imgs)):
        fw.write(imgs[i])
        fw.write('++')
        fw.write(nalls[i])
        fw.write('++')
        fw.write(corrs2[i])
        fw.write('++')
        fw.write(corrs3[i])
        fw.write('++')
        fw.write(corrs4[i])
        fw.write('++')
        fw.write(corrs5[i])
        # fw.write('\n')
    fw.close()
# with open(confall) as f:
#     for line in f:
#         img, fall = line.split('++')
#         imgs.append(img)
#         falls.append(fall.split('\n')[0])
#     f.close()
# with open(corr) as f:
#     for line in f:
#         caps = line.split('++')
#         corrs2.append(caps[2])
#         corrs3.append(caps[3])
#         corrs4.append(caps[4])
#         corrs5.append(caps[5])
#     f.close()
#
# with open(confall_corr2345, "a") as fw:
#     for i in range(len(imgs)):
#         fw.write(imgs[i])
#         fw.write('++')
#         fw.write(falls[i])
#         fw.write('++')
#         fw.write(corrs2[i])
#         fw.write('++')
#         fw.write(corrs3[i])
#         fw.write('++')
#         fw.write(corrs4[i])
#         fw.write('++')
#         fw.write(corrs5[i])
#         # fw.write('\n')
#     fw.close()

# with open(nothall_tmp) as f:
#     for line in f:
#         img, ll=line.split('++')
#         words=ll.split(' ')
#         with open(nothall, "a") as fw:
#             fw.write(img)
#             fw.write('++')
#             for i in range(len(words)-1):
#                 fw.write('nothing')
#                 fw.write(' ')
#             fw.write('nothing')
#             fw.write('\n')
#             fw.close()
#     f.close()

# with open(confall_tmp) as f:
#     for line in f:
#         img, ll=line.split('++')
#         words=ll.split(' ')
#         with open(confall, "a") as fw:
#             fw.write(img)
#             fw.write('++')
#             for i in range(len(words)-1):
#                 fw.write(words[0])
#                 fw.write(' ')
#             fw.write(words[0])
#             fw.write('\n')
#             fw.close()
#     f.close()



# #####create corr2345_conf_nothing
# corr2 = '../datasets/flickr30k/one_cap/flickr30k_test_corr2.txt'
# corr3 = '../datasets/flickr30k/one_cap/flickr30k_test_corr3.txt'
# corr4 = '../datasets/flickr30k/one_cap/flickr30k_test_corr4.txt'
# corr5 = '../datasets/flickr30k/one_cap/flickr30k_test_corr5.txt'
# # noth1 = '../datasets/flickr30k/one_cap/flickr30k_test_noth1.txt'
# # conf1 = '../datasets/flickr30k/one_cap/flickr30k_test_conf1.txt'
# # noth2 = '../datasets/flickr30k/one_cap/flickr30k_test_noth2.txt'
# # conf2 = '../datasets/flickr30k/one_cap/flickr30k_test_conf2.txt'
# nothall = '../datasets/flickr30k/one_cap/flickr30k_test_nothall.txt'
# confall = '../datasets/flickr30k/one_cap/flickr30k_test_confall.txt'
# corr2345_conf_noth = '../datasets/flickr30k/mix/nothall_confall_corr2345.txt'
# mix = []
# num_cap = 6
# for i in range(num_cap*1000):
#     mix.append([])
# i=0
# with open(nothall) as f:
#     for line in f:
#         mix[i].append(line)
#         i+=num_cap
#     f.close()
#
# i=1
# with open(confall) as f:
#     for line in f:
#         mix[i].append(line)
#         i+=num_cap
#     f.close()
# i=2
# with open(corr2) as f:
#     for line in f:
#         mix[i].append(line)
#         i+=num_cap
#     f.close()
# i=3
# with open(corr3) as f:
#     for line in f:
#         mix[i].append(line)
#         i+=num_cap
#     f.close()
# i=4
# with open(corr4) as f:
#     for line in f:
#         mix[i].append(line)
#         i+=num_cap
#     f.close()
#
# i=5
# with open(corr5) as f:
#     for line in f:
#         mix[i].append(line)
#         i+=num_cap
#     f.close()
#
#
#
# with open(corr2345_conf_noth, "a") as f:
#     for line in mix:
#         f.write(line[0])
#     f.close()

# #####test txt
# img_caps = []
# img_imgs = []
# ann_file = os.path.expanduser('../datasets/mscoco/captions_train.txt')
# with open(ann_file) as fd:
#     fd.readline()
#     for line in fd:
#         # if len(img_caps)==105:
#         #     a=1
#         if line:
#             # some lines have comma in the caption, se we make sure we do the split correctly
#             img_cap = line.split('++')
#             # if len(img_cap[0]) == 0:
#             #     a = 1
#             img_cap = int(img_cap[0])
#             img_caps.append(img_cap)
# ann_file = os.path.expanduser('../datasets/mscoco/img_names_train.txt')
# with open(ann_file) as fd:
#     fd.readline()
#     for line in fd:
#         line = line.strip()
#         if line:
#             # some lines have comma in the caption, se we make sure we do the split correctly
#             img_img = line.split('.jpg')
#             img_img = img_img[0].split('_0')
#             img_img = int(img_img[-1])
#             img_imgs.append(img_img)
# diff = np.array(img_imgs)-np.array(img_caps)
# diff=diff[diff!=0]
# a=1



# ###### process json
# ann_file = os.path.expanduser('../datasets/mscoco/coco_test_karpathy.json')
# with open(ann_file, 'r') as fp:
#     data = json.load(fp)
# imgs = data['images']
# annotations = data['annotations']
#
# captions = []
# img_names = []
# for i in range(581930):
#     captions.append([i])
# for i in range(581930):
#     img_names.append([])
#
# for capt in annotations:
#     image_id = capt['image_id']
#     cap = capt['caption']
#     captions[image_id].append(cap)
# for img in imgs:
#     id = img['id']
#     img_name = img['file_name']
#     img_names[id].append(img_name)
#
# img_names = [ele for ele in img_names if ele != []]
# captions = [ele for ele in captions if len(ele) != 1]
# tmp_test = [ele for ele in captions if len(ele) != 6]
# with open("C:/Users/INDA_HIWI/Desktop/captions.txt", "a") as f:
#     # a=0
#     for i in range(len(captions)):
#         caps = captions[i]
#         img_name = img_names[i]
#         log = str(img_name[0])+'++'+caps[1]+'++'+caps[2]+'++'+caps[3]+'++'+caps[4]+'++'+caps[5]
#         log=log.replace("\n", "")
#         log = log+'\n'
#         f.write(log)
#         # a+=1
#         # f.write(str(caps[0]))
#         # f.write('++')
#         # for i in range(5):
#         #     f.write(caps[i+1])
#         #     f.write('++')
#         # f.write('\n')
#     f.close()
