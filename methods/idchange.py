import os


# 将乱序的图片 以000，001，...顺序排列，以及将png和jpg格式的图片都转换为png格式
class BatchRename():

    def rename(self):

        path = "H:/DATASET/TJML"  # 存放图片的路径
        filelist = os.listdir(path)
        total_num = len(filelist)
        i = 1
        for item in filelist:
            if item.endswith('.jpeg') or item.endswith('.png') or item.endswith('.json') or item.endswith('.jpg') or item.endswith('.bmp'):  # 判断图片格式
                src = os.path.join(os.path.abspath(path), item)

                # 此处可自己修改，换成想要的排序方式
                if i < 10:
                    dst = os.path.join(os.path.abspath(path), 'img' + '00' + str(i) + '.png')
                elif i <=99:
                    dst = os.path.join(os.path.abspath(path), 'img' + '0' +str(i) + '.png')
                elif i >= 100:
                    dst = os.path.join(os.path.abspath(path), 'img' + str(i) + '.png')

                try:
                    os.rename(src, dst)
                    i += 1
                except:
                    continue
        print('total %d to rename & converted %d png' % (total_num, i-1))


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()


# import os
#
# # 将乱序的图片 以000，001，...顺序排列，以及将png和jpg格式的图片都转换为png格式
# class BatchRename():
#     def rename(self):
#         path = "E:/Line/cable/train/type1"  # 存放图片的路径
#         filelist = os.listdir(path)
#         total_num = len(filelist)
#         i = 1
#         for item in filelist:
#             if item.endswith('.jpg') or item.endswith('.png'):  # 判断图片格式
#                 src = os.path.join(os.path.abspath(path), item)
#                 # 根据编号位数生成目标文件名
#                 if i < 10:
#                     dst = os.path.join(os.path.abspath(path), 'fl00' + str(i) + '.png')
#                 elif i < 100:
#                     dst = os.path.join(os.path.abspath(path), 'fl0' + str(i) + '.png')
#                 else:
#                     dst = os.path.join(os.path.abspath(path), 'fl' + str(i) + '.png')
#
#                 try:
#                     os.rename(src, dst)
#                     i += 1
#                 except:
#                     continue
#         print('total %d to rename & converted %d png' % (total_num, i - 1))
#
# if __name__ == '__main__':
#     demo = BatchRename()
#     demo.rename()