预期格式
```
┕━data
    ┠─birds
        ┠─Black_footed_Albatross
        ┠─Crested_Auklet
        ┠─Groove_billed_Ani
        ┠─Indigo_Bunting
        ┗━Laysan_Albatross
    ┗━split_dataset.py
```
在`data`文件夹下启动`cmd`，输入以下命令
```cmd
python split_dataset.py --data_path=birds --split_rate=0.1
```
生成如下文件结构
```
┕━data
    ┠─birds
        ┠─Black_footed_Albatross
        ┠─Crested_Auklet
        ┠─Groove_billed_Ani
        ┠─Indigo_Bunting
        ┗━Laysan_Albatross
    ┠─train
        ┠─Black_footed_Albatross
        ┠─Crested_Auklet
        ┠─Groove_billed_Ani
        ┠─Indigo_Bunting
        ┗━Laysan_Albatross
    ┠─test
        ┠─Black_footed_Albatross
        ┠─Crested_Auklet
        ┠─Groove_billed_Ani
        ┠─Indigo_Bunting
        ┗━Laysan_Albatross
    ┗━split_dataset.py
```

> 代码[参考](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/data_set/split_data.py)