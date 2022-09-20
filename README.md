# Face_Alignment
>***将图像进行人脸对齐***

## 说明

该项目依赖pytorch框架，具体环境依赖、安装说明以及教程见[INSTALL.md](./INSTALL.md)

## face_alignment
```bash
代码均在该目录下
```
### image_example
- 里面存放了人脸对齐前后的对比
### alignment.*
- 这两个文件是单张照片对齐的实现
### face_alignment.py
- 该文件实现了批量文件的人脸对齐
- 其实现的效果是在照片文件夹所在的目录下生成一个额外的文件夹，里面的目录结构跟源文件夹一模一样，图像进行了对齐。
- 文件夹的格式要求是：同一类别的照片用一个子文件夹存储，子文件夹的命名为类别的标签名称
```python
#在此处设置文件夹路径然后运行即可
source_dir = '/home/fxf/my_model/resnet50/resnet50_test/test_images'
```
