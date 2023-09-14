import os
from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
from utils import load_model_on_gpus
import re
import torch
from docx import Document
import time
import PyPDF2

ARTICLE_PROMPT_FOLDER = 'article_summary_prompts'
REPORT_PROMPT_FOLDER = 'report_summary_prompts'

import requests
from bs4 import BeautifulSoup
import io

def process_uploaded_files(file_wrappers):
    """
    Convert uploaded files into text.
    """
    content_list = []

    # Check if a single file is uploaded or multiple files
    if not isinstance(file_wrappers, list):
        file_wrappers = [file_wrappers]

    for file_wrapper in file_wrappers:
        file_name = file_wrapper.name
        file_content = file_wrapper.read()

        if file_name.endswith(".doc") or file_name.endswith(".docx"):
            print (f"++++ Reading Word document {file_name}.")
            # doc = Document(io.BytesIO(file_content))
            doc = Document(file_name)
            content = "\n".join([para.text for para in doc.paragraphs])
        elif file_name.endswith(".pdf"):
            print(f"++++ Reading PDF document {file_name}.")
            with open(file_name, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = "\n".join([reader.pages[i].extract_text() for i in range(len(reader.pages))])
        elif file_name.endswith(".html"):
            print (f"++++ Reading and processing HTML document {file_name}...")
            soup = BeautifulSoup(file_content, 'lxml')
            for script in soup(["script", "style"]):
                script.decompose()
            content = " ".join(soup.stripped_strings)
        else:
            print (f"++++ Reading plain text document {file_name}...")
            content = file_content.decode('utf-8')

        # Wrap content
        wrapped_content = f"\n\n====上傳文檔開始(file://{file_name})===={{\n{content}\n}}====上傳文檔結束(file://{file_name})====\n"
        content_list.append(wrapped_content)

    return "\n".join(content_list)



def download_and_clean_html(url):
    try:
        print(f"++++ Downloading content from {url}...")
        response = requests.get(url)
        response.raise_for_status()

        print(f"++++ Cleaning up content from {url}...")
        soup = BeautifulSoup(response.text, 'lxml')
        # Remove unnecessary tags
        for script in soup(["script", "style"]):
            script.decompose()
        # Extract the text and clean it up
        return " ".join(soup.stripped_strings)
    except requests.RequestException as e:
        print(f"++++ Error fetching content from {url}: {e}")
        return f"Error fetching content from {url}: {e}"

def generate_report(article_summary_prompt, report_summary_prompt, urls, chatbot, max_length, top_p, temperature, history, past_key_values):
    urls = urls.split("\n")
    article_summaries = []

    summary_count = 1

    # 1. and 2. Download each article's content, concatenate it with the chosen article summary prompt
    for url in urls:
        content = download_and_clean_html(url)
        if not content:
            print(f"Skipping {url} due to download error or empty content.")
            continue
        article_input = f"{buildPromptFileFullPath(ARTICLE_PROMPT_FOLDER, article_summary_prompt)} {content}"

        # Use the model to generate a summary
        summary_response, new_history, new_past_key_values = predict(
            article_input, chatbot, max_length, top_p, temperature, history, past_key_values
        )
        if summary_response:
            summary = summary_response[-1][1]  # Take the last response (the generated summary)
            
            # Prepend the string with an incremental count
            formatted_summary = f"\n文章摘要 {summary_count}:\n{summary}"
            article_summaries.append(formatted_summary)
            
            # Increment the counter
            summary_count += 1
            
            history = new_history  # Update the history for the next iteration
            past_key_values = new_past_key_values  # Update past_key_values for the next iteration

    # 3. Collect all summaries and concatenate them to the reportSummaryPrompt
    combined_input = f"{buildPromptFileFullPath(REPORT_PROMPT_FOLDER, report_summary_prompt)} {' '.join(article_summaries)}"

    # Generate the final report
    report_response, _, _ = predict(
        combined_input, chatbot, max_length, top_p, temperature, history, past_key_values
    )
    if report_response:
        report = report_response[-1][1]  # Take the last response (the generated report)
        return report
    return "Error generating report"

def load_prompts_from_folder(folder_path):
    files_in_folder = os.listdir(folder_path)
    print(f"Files in {folder_path}: {files_in_folder}")
    return [file[:-4] for file in files_in_folder if file.endswith('.txt')]

def buildPromptFileFullPath(folder_path, prompt_file_name):
    return os.path.join(folder_path, prompt_file_name + '.txt')

print(f"Current Working Directory: {os.getcwd()}")
article_summary_prompts = load_prompts_from_folder(ARTICLE_PROMPT_FOLDER)
report_summary_prompts = load_prompts_from_folder(REPORT_PROMPT_FOLDER)

print(f"Article Summary Prompts: {article_summary_prompts}")
print(f"Report Summary Prompts: {report_summary_prompts}")

if torch.cuda.is_available():
    model_path = "../chatglm2-6b"
    print (f"CUDA is available. Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_model_on_gpus(model_path, num_gpus=2)
else: 
    model_path = "THUDM/chatglm2-6b"
    print (f"CUDA is not available. Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to('mps')
    
# model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().to('mps')
# tokenizer = AutoTokenizer.from_pretrained("../chatglm2-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("../chatglm2-6b", trust_remote_code=True).cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("../chatglm2-6b", num_gpus=2)
model = model.eval()

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def find_urls(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(text)


def predict(input, uploaded_files, chatbot, max_length, top_p, temperature, history, past_key_values, downloadUrlContent=False):
    if uploaded_files:
        file_contents = process_uploaded_files(uploaded_files)
        input += "\n" + file_contents
        
    if downloadUrlContent:
        # 1. Find all URLs in the input text
        urls = find_urls(input)

        # 2. Download the content of each URL and replace it in the input text
        for url in urls:
            content = download_and_clean_html(url)
            print (f"++++ Downloaded content from {url}: {content}")

            input = input.replace(url, "\n\n====網頁內容開始(" + url + ")===={\n\n" + content + "\n\n}====網頁內容結束(" + url + ")====\n")

    chatbot.append((parse_text(input), ""))

    print(f"++++ Predicting started. input_length={len(input)}")

    # Get the current time before the loop starts
    start_time = time.time()

    for response, history, past_key_values in model.stream_chat(tokenizer, input, history, past_key_values=past_key_values,
                                                                return_past_key_values=True,
                                                                max_length=max_length, top_p=top_p,
                                                                temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))
        yield chatbot, history, past_key_values

    # Calculate the time difference after the loop completes
    end_time = time.time()
    duration = end_time - start_time

    print(f"++++ Prediction completed in {duration:.2f} seconds. input_length={len(input)} and response_length={len(response)}")


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], None

# sampleInput = """設想你是富蘭克林基金公司的研究員，請將以下金融文章的網頁內容總結成600字以內的摘要，請務必簡潔準確客觀。每一篇網頁內容都包括在其“網頁內容開始”和“網頁內容結束”之間。

# https://www.cnbc.com/2023/09/05/bond-yield-jump-is-not-death-to-equities-bofas-savita-subramanian.html
# """
sampleInput = """假設你是富蘭克林基金的研究員。請從以下金融文章中摘取要點，並將其總結為不超過1000字的報告，以供我們的客戶參考。請確保報告簡潔、準確且客觀。回答請使用繁體中文。如有參考資料或鏈接，請在報告末尾列出。

當您看到“網頁內容開始”和“網頁內容結束”時，其間的內容為網頁文章；當您看到“上傳文檔開始”和“上傳文檔結束”時，其間的內容為上傳的文件內容。

https://www.cnbc.com/2023/09/05/bond-yield-jump-is-not-death-to-equities-bofas-savita-subramanian.html

"""

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChifuGPT v0.1</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Column(scale=4):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=12).style(container=False)
            with gr.Column(scale=1):
                file_uploads = gr.Files(label="上傳本地Word/PDF/HTML文檔以嵌入提示", file_count="multiple", height=100)
                downloadUrlContent = gr.Checkbox(label="下載並嵌入提示中的http鏈接文章", value=True)
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=2): # Adjust the 'scale' as per your design
            article_prompt_selector = gr.Dropdown(choices=article_summary_prompts, label="Choose article-summary prompt")
            report_prompt_selector = gr.Dropdown(choices=report_summary_prompts, label="Choose report-summary prompt")
            
            article_urls = gr.Textbox(label="引用文章鏈接列表", lines=5, placeholder="Enter URLs (one per line)", container=False)
            generate_report_btn = gr.Button("Generate Report")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    user_input.value = sampleInput
    
    history = gr.State([])
    past_key_values = gr.State(None)

    submitBtn.click(predict, [user_input, file_uploads, chatbot, max_length, top_p, temperature, history, past_key_values, downloadUrlContent],
                    [chatbot, history, past_key_values], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

    generate_report_btn.click(
        generate_report,
        [article_prompt_selector, report_prompt_selector, article_urls, chatbot, max_length, top_p, temperature, history, past_key_values],
        outputs=[gr.Textbox(label="Generated Report")]
    )

if torch.cuda.is_available():
    listen_ip = "0.0.0.0"
    listen_port = 7860
else:
    listen_ip = "127.0.0.1"
    listen_port = 8081

demo.queue().launch(server_name=listen_ip, server_port=listen_port, share=False, inbrowser=False)
