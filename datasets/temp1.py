import os
ann_file = '../datasets/mscoco/coco_test1k.txt'
coco_ori = '../datasets/mscoco/one_cap/coco_test_ori1.txt'
# confall_path = 'C:/Users/INDA_HIWI/Desktop/ff/flickr30k_test_confall.txt'
# conf1='../datasets/flickr30k/one_cap/flickr30k_test_conf1.txt'
# conf2='../datasets/flickr30k/one_cap/flickr30k_test_conf2_new.txt'
# noth2='../datasets/flickr30k/one_cap/flickr30k_test_noth2.txt'
# conf2_corr2345='../datasets/flickr30k/mix/conf2_corr2345_new.txt'
# conf1_conf2_corr2345='../datasets/flickr30k/mix/conf1_conf2_corr2345_new.txt'
# noth2_conf2_corr2345='../datasets/flickr30k/mix/noth2_conf2_corr2345_new.txt'
# # noth2_path = 'C:/Users/INDA_HIWI/Desktop/ff/flickr30k_test_noth2.txt'
# # corr1_path = '../datasets/flickr30k/one_cap/flickr30k_test_corr1.txt'
# corr2_path = '../datasets/flickr30k/one_cap/flickr30k_test_corr2.txt'
# corr3_path = '../datasets/flickr30k/one_cap/flickr30k_test_corr3.txt'
# corr4_path = '../datasets/flickr30k/one_cap/flickr30k_test_corr4.txt'
# corr5_path = '../datasets/flickr30k/one_cap/flickr30k_test_corr5.txt'
# ann_file = 'C:/Users/INDA_HIWI/Desktop/coco_test5k.txt'
# coco1k = 'C:/Users/INDA_HIWI/Desktop/coco_test1k.txt'

# with open(ann_file) as fd:
#     for line in fd:
#         cap_conf = []
#         img, cap_0 = line.split('.jpg,')
#         img = img + '.jpg,'
#         cap_conf.append(img)
#
#         cap_1 = cap_0.split(',')
#         gt = cap_1[0].split(' ')[0] + ' '
#         for items in cap_1:
#             cap_2 = items.split(' ')
#             for word in cap_2:
#                 cap_conf.append(gt)
#             cap_conf.append(', ')
#         cap_conf.append('.\n')
#         with open(confall_path, "a") as f:
#             for w in cap_conf:
#                 f.write(w)
#             f.close()



with open(ann_file) as fd:
    with open(coco_ori, "a") as f:
        for line in fd:
            cap_ori=line.split('++')[1]
            f.write(cap_ori)
            f.write('\n')
        f.close()
# conf1_cap=[]
# noth2_cap=[]
# conf2_cap=[]
# corr2_cap=[]
# corr3_cap=[]
# corr4_cap=[]
# corr5_cap=[]
# with open(noth2) as fd:
#     for line in fd:
#         noth2_cap.append(line)
# with open(conf2) as fd:
#     for line in fd:
#         conf2_cap.append(line)
# with open(corr2_path) as fd:
#     for line in fd:
#         corr2_cap.append(line)
# with open(corr3_path) as fd:
#     for line in fd:
#         corr3_cap.append(line)
# with open(corr4_path) as fd:
#     for line in fd:
#         corr4_cap.append(line)
# with open(corr5_path) as fd:
#     for line in fd:
#         corr5_cap.append(line)
#
#
# with open(noth2_conf2_corr2345, "a") as f:
#     for i in range(1000):
#         f.write(noth2_cap[i])
#         f.write(conf2_cap[i])
#         # f.write('\n')
#         f.write(corr2_cap[i])
#         # f.write('\n')
#         f.write(corr3_cap[i])
#         # f.write('\n')
#         f.write(corr4_cap[i])
#         # f.write('\n')
#         f.write(corr5_cap[i])
#         # f.write('\n')
#     f.close()




# with open(ann_file) as fd:
#     fd.readline()
#     i=0
#     for line in fd:
#         i_cap=(i%10)
#
#         if i_cap==1:
#             with open(conf2, "a") as f:
#                 f.write(line)
#                 f.close()
#         elif i_cap==1:
#             with open(corr2_path, "a") as f:
#                 f.write(line)
#                 f.close()
#         elif i_cap == 2:
#             with open(corr3_path, "a") as f:
#                 f.write(line)
#                 f.close()
#         elif i_cap == 3:
#             with open(corr4_path, "a") as f:
#                 f.write(line)
#                 f.close()
#         elif i_cap == 4:
#             with open(corr5_path, "a") as f:
#                 f.write(line)
#                 f.close()
#         else:
#             print('error!')
#
#         i+=1






























#
#
# ann_file='C:/Users/INDA_HIWI/Desktop/MasterThesis/datasets/mscoco/temp/coco_test1k.txt'
# # ann_file='C:/Users/INDA_HIWI/Desktop/MasterThesis/datasets/mscoco/temp/coco_test1k - Noth.txt'
# # ann_file='C:/Users/INDA_HIWI/Desktop/MasterThesis/datasets/mscoco/temp/coco_test1k - Noth_2.txt'
# # ann_file='C:/Users/INDA_HIWI/Desktop/MasterThesis/datasets/mscoco/temp/coco_test1k_2.txt'
# conf1 = '../datasets/mscoco/one_cap/coco_test_conf1.txt'
# conf2 = '../datasets/mscoco/one_cap/coco_test_conf2.txt'
# noth1 = '../datasets/mscoco/one_cap/coco_test_noth1.txt'
# noth2 = '../datasets/mscoco/one_cap/coco_test_noth2.txt'
#
# # with open(ann_file) as fd:
# #     for line in fd:
# #         data =line.split('++')
# #         img_id = data[0]
# #
# #         i = 1
# #         with open(conf2, "a") as f:
# #             f.write(img_id)
# #             f.write('++')
# #             f.write(data[i])
# #             f.write('\n')
# #             f.close()
# #     fd.close()
#
#
# img_id=[]
# cap_1=[]
# cap_2=[]
# ori_1=[]
# ori_2=[]
# ori_3=[]
# ori_4=[]
# conf1_conf2_corr2345='../datasets/mscoco/mix/conf1_conf2_corr2345.txt'
# noth1_conf1_corr2345='../datasets/mscoco/mix/noth1_conf1_corr2345.txt'
# noth1_conf2_corr2345='../datasets/mscoco/mix/noth1_conf2_corr2345.txt'
# noth1_noth2_corr2345='../datasets/mscoco/mix/noth1_noth2_corr2345.txt'
# noth2_conf2_corr2345='../datasets/mscoco/mix/noth2_conf2_corr2345.txt'
#
# with open(ann_file) as fd:
#     for line in fd:
#         data =line.split('++')
#         img_id.append(data[0])
#         ori_1.append(data[2])
#         ori_2.append(data[3])
#         ori_3.append(data[4])
#         ori_4.append(data[5].split('\n')[0])
#     fd.close()
#
# with open(noth2) as fd:
#     for line in fd:
#         data =line.split('++')
#         cap_1.append(data[1].split('\n')[0])
#     fd.close()
# with open(conf2) as fd:
#     for line in fd:
#         data =line.split('++')
#         cap_2.append(data[1].split('\n')[0])
#     fd.close()
#
# with open(noth2_conf2_corr2345, "a") as f:
#     for i in range(1000):
#         f.write(img_id[i])
#         f.write('++')
#         f.write(cap_1[i])
#         f.write('++')
#         f.write(cap_2[i])
#         f.write('++')
#         f.write(ori_1[i])
#         f.write('++')
#         f.write(ori_2[i])
#         f.write('++')
#         f.write(ori_3[i])
#         f.write('++')
#         f.write(ori_4[i])
#         f.write('\n')
#     f.close()