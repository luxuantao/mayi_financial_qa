# 算法思路

使用retrieval中的search_qa部分进行模型训练

模型使用 `mt5-base` 进行端到端训练

模型输入为：原问题 + '<|question|>' + search_qa['question'] + '<|answer|>' + search_qa['answer']

输出为相应的问答

其中 '<|question|>' 和 '<|answer|>' 为添加的特殊字符



# 复现流程

1. `pip install -r requirements.txt`

2. 下载比赛数据到 `data` 目录下

3. `python read_data.py` 预处理数据

4. `python train.py` 模型训练

5. `python inference.py` 模型预测，会在当前工作目录下生成 `final.csv` 文件，即为最终预测结果

   

   

   

   

   

   