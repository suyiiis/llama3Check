# 使用pipeline加载模型
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import json
from datasets import load_dataset, Dataset
import tqdm
from huggingface_hub import login

login("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")
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
save_path = "identify_result.json"
file_path = '../whoishp/llama2GeneratedText_hp.json'
batch_size = 256
torch.cuda.empty_cache()
generator = pipeline("text-generation",
                     model=model_path,
                     tokenizer=model_path,
                     model_kwargs={"torch_dtype": torch.bfloat16},
                     device=7)
generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
generator.tokenizer.padding_side = "left"
print("model loaded")
# read file to var hp_text
with open(file_path, 'r', encoding='utf-8') as f:
    hp_text = f.read().split('\n')
prompt_list = []
# prompt_list = [json.loads(t)['generated'].replace(json.loads(t)['raw_text'], '') for t in hp_text]
for t in hp_text:
    if len(t) == 0:
        break
    tmp_json = json.loads(t)
    pmt = tmp_json['generated'].replace(tmp_json['raw_text'], '')
    messages = [{"role": "system",
                 "content": "You are a robot that recognizes whether a given text is related to Harry Potter, and if so you only need to answer yes. If it's irrelevant, you just need to answer no."},
                {"role": "user",
                 "content": pmt}]
    prompt = generator.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_list.append(prompt)


def data(all_text):
    for text in all_text:
        yield text


def run():
    all_testfile = open(save_path, 'a')
    cnt = 0
    terminators = [
        generator.tokenizer.eos_token_id,
        generator.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    try:
        for res in tqdm.tqdm(generator(data(prompt_list),
                                       max_new_tokens=10,
                                       eos_token_id=terminators,
                                       batch_size=batch_size,
                                       do_sample=False,
                                       ),
                             desc='main1:'):
            user_input = prompt_list[cnt]
            tmp = res[0]['generated_text'][len(prompt_list[cnt]):]
            to_save = {
                'No.': cnt,
                'R': tmp,
            }
            json.dump(to_save, all_testfile)
            all_testfile.write("\n")
            torch.cuda.empty_cache()
            cnt += 1
            all_testfile.flush()
        all_testfile.close()
    except Exception as e:
        # 处理 generator 异常，可以打印错误信息或采取其他措施
        print(f"Generator error: {e}")
        # 如果需要继续处理下一个输入，可以增加 cnt
        time.sleep(5)


def send_mail_with_attachment(file_path):
    import smtplib
    # email 用于构建邮件内容
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    from email.mime.multipart import MIMEMultipart
    from email.mime.application import MIMEApplication

    # 构建邮件头
    from email.header import Header
    # 发信方的信息：发信邮箱，QQ 邮箱授权码
    from_addr = '1225748424@qq.com'  # 发生者的邮箱  您的qq邮箱
    password = 'ruajmzwktxhsjjhc'  # 刚才短信获取到的授权码
    # 收信方邮箱
    to_addr = '1225748424@qq.com'
    # 发信服务器
    smtp_server = 'smtp.qq.com'

    content = '实验完毕'
    textApart = MIMEText(content)
    textApart['From'] = Header('1225748424@qq.com')  # 发送者
    textApart['To'] = Header('nin')  # 接收者
    subject = 'Python SMTP 邮件测试'  # 主题
    textApart['Subject'] = Header(subject, 'utf-8')  # 邮件主题

    pdfFile = file_path
    pdfApart = MIMEApplication(open(pdfFile, 'rb').read())
    pdfApart.add_header('Content-Disposition', 'attachment', filename=pdfFile)

    # zipFile = '算法设计与分析基础第3版PDF.zip'
    # zipApart = MIMEApplication(open(zipFile, 'rb').read())
    # zipApart.add_header('Content-Disposition', 'attachment', filename=zipFile)

    m = MIMEMultipart()
    m.attach(textApart)
    m.attach(pdfApart)
    # m.attach(zipApart)
    m['Subject'] = '1225748424@qq.com'
    m['From'] = Header('1225748424@qq.com')
    try:
        server = smtplib.SMTP('smtp.qq.com')
        server.login(from_addr, password)
        server.sendmail(from_addr, to_addr, m.as_string())
        print('success')
        server.quit()
    except smtplib.SMTPException as e:
        print('error:', e)  # 打印错误


run()
send_mail_with_attachment(save_path)
