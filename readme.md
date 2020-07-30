### 证件顶点回归模型(Regression Four Vertex of Card)

#### data
* 图片：长方形证件
* 标注内容： 证件四个角点坐标

#### train
* python train.py
* 得到的模型能够定位证件四个点

#### inference
* 过模型得到四个角点
* 利用cv2.perspective进行透视变换回正证件