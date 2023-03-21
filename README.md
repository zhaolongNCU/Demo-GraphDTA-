# Demo-GraphDTA-
小白入门DTA方向，GraphDTA作为经典的DTA模型，对于python基础不好，深度学习代码实操不强者，本github将代码细致的进行注释阐述，旨在记录学习过程，帮助更多入门者尽快入门！
# 资源：

- README.md：此文件
- GraphDTA本身代码可以从这里https://github.com/thinng/GraphDTA进行下载

## 源代码解释文件

- create_data.py：以pytorch格式创建数据
- utils.py：包括create_data.py用于创建数据的TestbedDataset类和评估函数mse、ci等
- trainging.py：训练GraphDTA模型
- models/ginconv.py、gat.py、gat_gcn.py和 gcn.py：文章提出的四个图模型框架
- 这里对每个文件代码进行了较为详细的注释，见各文件代码笔记.ipynb

# 复现步骤

## 1.安装所需python库

```
conda create -n geometric python=3
conda activate geometric
conda install -y -c conda-forge rdkit
conda install pytorch torchvision cudatoolkit -c pytorch
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
```

可以直接通过如上指令进行配置环境，注意这里配置可能会出现版本不匹配问题，面向百度搜索解决下就OK

## 2.以pytorch格式创建数据

```
conda activate geometric 
python create_data.py
```

首先激活你所配置的环境，再运行create_data.py文件，这一步操作完成后会生成相应数据集的训练、测试的csv文件。并其这些数据被以pytorch格式数据（方便调用和存储）存储在data/processing中，由 kiba_train.pt、kiba_test.pt、davis_train.pt 和 davis_test.pt 组成。

具体可以理解这个文件的代码笔记，可以更清楚的理解怎么进行数据预处理得到想要的药物和蛋白质的表征。

## 3.训练预测模型

使用训练数据训练模型。如果模型获得用于测试数据的最佳 MSE，则选择该模型。
运行

```python
conda activate geometric
python training.py 0 0 0
```

这里主要理解这三个外部参数，由于在training.py文件中运用了`sys.argv[]`函数，这个函数可以导入外部参数，这里就是外部参数。其实这里包括四个外部参数，分别时training.py、0、0、0，第一个参数没有用处我们不管，具体看其他三个参数，第一个0对应的是选择哪个数据集Davis、KIBA。这里可以返回training.py文件寻找`sys.argv[1]`理解那里的这个设置就理解了为什么这里第一个0用来选择数据集，因为training.py文件就是通过这个参数的获取来选择数据集的0对应Davis，1对应KIBA数据集。

以此类推通过这样方式理解第二个0对应的四个模型的选择，分别为 GINConvNet、GATNet、GAT_GCN 、 GCNNet 的 0/1/2/3。

第三个0是cuda的选择，即GPU的选择，0/1 是 'cuda：0'或“cuda：1”

比如你想利用cuda：1，采用GATNet模型在KIBA数据上训练，那么运行如下代码即可：

```python
python training.py 1 1 1
```

最后返回建模的模型和结果文件，以实现在整个训练过程中测试数据的最佳 MSE。 例如，在Davis数据上运行 GATNet 时，它会返回两个文件 model_GATNet_davis.model 和 result_GATNet_davis.csv。

详细理解可以参考代码笔记文件

## 4.通过验证集训练预测模型

运行

```
python training_validation.py 0 0 0
```

参数同上理解即可，这里和刚刚的区别在于我们是否分割训练集为训练和验证集，并拿验证集上最好的mse模型来预测测试集得到相应地五个评估指标。这遵循了[在 https://github.com/hkmztrk/DeepDTA](https://github.com/hkmztrk/DeepDTA) 中选择模型的方式。不过，两种训练方式的结果是可比的。

这遵循了[在 https://github.com/hkmztrk/DeepDTA](https://github.com/hkmztrk/DeepDTA) 中选择模型的方式。不过，两种训练方式的结果是可比的。

详细理解可以参考代码笔记文件
