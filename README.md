# Unique-hackday_solvegpt

# 软件截图
![图片](https://github.com/hongshen-zhang/Unique-hackday_solvegpt/assets/51727955/07e774e9-1c41-4daf-9eed-44c4673b5169)


# 产品功能:
1. 多模态输入(Multimodal Input)，用于识别图片题目公式与文字(基于腾讯云OCR)
2. 多模型联合对抗求解(Adversarial Learning)，用于提高求解准确率(gpt-3.5-turbo,gpt-4,gpt-4-0613)
3. 准确率智能判断(Accuracy Analysis)，用于评估解题答案的准确性(Accuracy 0% - 100%)
4. 跨平台部署(Cross-platform)，支持Windows，Mac以及Android设备
5. 题库学习(Bank Learning),直接输入题目或者公式进行学习


# 开发总结

1. 网页端: 开发了 [solvegpt](http://118.89.117.111/solvegpt/index.html)
2. 安卓端: 开发了 Android端App Solvegpt

---


# 代码运行说明

```
:: 1. ocr secret key
./solvegpt/main.py line 95:
def tencent_ocr(img_base64):
    cred = credential.Credential(
        "", ""
    )
 
:: 2. openai secret key
./solvegpt/openai_config.json line 2:
{
    "api_key": "",
