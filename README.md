# wbmatting
White background matting

## 使用方式

### 安装依赖
```
python -m pip install -r requirements.txt
```


```
python wbmatting.py \
    --model torchscript.pth  \
    --src testdata/6900068804425.jpg \
    --device [cpu|cuda] \
    --output-dir output
```