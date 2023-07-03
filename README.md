![图片](https://github.com/hongshen-zhang/Unique-hackday_solvegpt/assets/51727955/4fb40777-9f97-4aca-9067-3c40cf78c394)# AIGC + Accessible

# 背景介绍
目前，一种主流的观点是:chatgpt能帮我们解决作业问题(90%的国外学生都用chatgpt写作业)。
而实际上，目前chatgpt(尤其是理科问题)并不accessible。
一个做题的流程是，1.打开vpn。2.登陆openai。3.将文字和公式输入对话框。4.得到一个不太准确的结果。
整个过程存在许多问题
1. 网络限制: 需要连接外网。
2. 账号限制: 需要注册海外账号。
3. 输入复杂: 手动将文字和公式输入chatgpt。
4. 结果不准：chatgpt目前在解决理科题目时，准确率低。(高考理科填空题和大题准确率不足20%)

因此,我们开发了solvegpt，一款AI实时集智拍照解题工具

1. 无网络限制: 部署在腾讯云(使用香港节点中转)。
2. 无账号限制: 我们目前免费开放使用(我垫钱)。
3. 输入简单: 支持识别图片题目公式与文字。
4. 结果判断：基于多模型(gpt3.5,4)联合讨论，我们能够更准确的求解(相比于直接求解)。对于实在无法解决的问题，我们输出了准确率的概率，给学生提供参考。

# 软件截图
![图片](https://github.com/hongshen-zhang/Unique-hackday_solvegpt/assets/51727955/3271fb97-5f40-4f7d-8d2b-54baf6908701)

# 产品功能:
1. 多模态输入(Multimodal Input)，用于识别图片题目公式与文字(基于腾讯云OCR)
2. 多模型联合对抗求解(Adversarial Learning)，用于提高求解准确率(gpt-3.5-turbo,gpt-4,gpt-4-0613)
3. 准确率智能判断(Accuracy Analysis)，用于评估解题答案的准确性(Accuracy 0% - 100%)
4. 跨平台部署(Cross-platform)，支持Windows，Mac以及Android设备
5. 题库学习(Bank Learning),直接输入题目或者公式进行学习


# 软件部署

1. 网页端: 开发了 [solvegpt](http://118.89.117.111/solvegpt/index.html)
2. 安卓端: 开发了 Android端App Solvegpt

---


# 代码运行
请补充OCR Key和OPENAI Key

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
