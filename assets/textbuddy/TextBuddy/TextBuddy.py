# TextBuddy v1.0 - 你的便捷文本分析工具
import os
import requests
from bs4 import BeautifulSoup
import string
import pandas as pd
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import TreebankWordTokenizer
from docx import Document
from docx.shared import Inches
import re


# Complete English text from Grimms' Fairy Tales, Project Gutenberg #2591.
# Collection by Jacob and Wilhelm Grimm; English text based on translations
# by Edgar Taylor and Marian Edwardes. No abridgment or rewritten wording.
SAMPLE_TITLE = "The Fox and the Cat"
SAMPLE_SOURCE = "https://www.gutenberg.org/files/2591/2591-h/2591-h.htm"
SAMPLE_TEXT = """It happened that the cat met the fox in a forest, and as she thought to herself: ‘He is clever and full of experience, and much esteemed in the world,’ she spoke to him in a friendly way. ‘Good day, dear Mr Fox, how are you? How is all with you? How are you getting on in these hard times?’ The fox, full of all kinds of arrogance, looked at the cat from head to foot, and for a long time did not know whether he would give any answer or not.

At last he said: ‘Oh, you wretched beard-cleaner, you piebald fool, you hungry mouse-hunter, what can you be thinking of? Have you the cheek to ask how I am getting on? What have you learnt? How many arts do you understand?’ ‘I understand but one,’ replied the cat, modestly. ‘What art is that?’ asked the fox. ‘When the hounds are following me, I can spring into a tree and save myself.’ ‘Is that all?’ said the fox. ‘I am master of a hundred arts, and have into the bargain a sackful of cunning.

You make me sorry for you; come with me, I will teach you how people get away from the hounds.’ Just then came a hunter with four dogs. The cat sprang nimbly up a tree, and sat down at the top of it, where the branches and foliage quite concealed her. ‘Open your sack, Mr Fox, open your sack,’ cried the cat to him, but the dogs had already seized him, and were holding him fast. ‘Ah, Mr Fox,’ cried the cat.

‘You with your hundred arts are left in the lurch! Had you been able to climb like me, you would not have lost your life.’"""

# --- 用户可配置参数 ---
# 在这里设置您希望在Word文档标题中使用的字体名称。
# 请确保您的电脑上已安装此字体，否则Word将使用默认字体。
# 示例: "Times New Roman", "Arial", "Calibri", "宋体"
HEADING_FONT_NAME = "Times New Roman"


# --- 核心分析功能模块 ---

def tokenize_words(text):
    """English word tokens; preserve internal apostrophes, split hyphens."""
    normalized = text.lower().replace("’", "'").replace("‘", "'")
    return re.findall(r"[a-z]+(?:'[a-z]+)*", normalized)


def clean_text(text):
    return " ".join(tokenize_words(text))


def calculate_ttr(text):
    words = tokenize_words(text)
    return len(set(words)) / len(words) if words else 0.0


def _ld_tagged_tokens(text):
    """Tokenize English sentences before POS tagging; retain case and context.

    Treebank tokenization splits contractions and punctuation. LD uses word
    tokens (including contracted word parts), not punctuation or numbers.
    """
    normalized = text.replace("’", "'").replace("‘", "'")
    tokenizer = TreebankWordTokenizer()
    tagged = []
    try:
        for sentence in re.split(r'(?<=[.!?])\s+', normalized):
            tokens = tokenizer.tokenize(sentence)
            if tokens:
                tagged.extend(pos_tag(tokens, lang='eng'))
    except LookupError as error:
        raise RuntimeError(
            "LD needs the NLTK English POS tagger. Install it once with: "
            "python -m nltk.downloader averaged_perceptron_tagger_eng "
            "(older NLTK releases may require averaged_perceptron_tagger)."
        ) from error
    return tagged


def _is_ld_word(token):
    return bool(re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?|'[A-Za-z]+", token))


def _is_ld_content(tagged, index):
    """Operational LD definition: nouns, lexical verbs, adjectives, adverbs.

    All forms of be, negation not/n't, and common have/do auxiliaries are
    excluded. Auxiliary identification is a local POS-based heuristic, not
    a syntactic parser; automatic tagging and ambiguous usages can vary.
    """
    token, tag = tagged[index]
    word = token.lower()
    if not _is_ld_word(token):
        return False
    if tag.startswith(('NN', 'JJ')):
        return True
    if tag.startswith('RB'):
        return word not in {'not', "n't"}
    if not tag.startswith('VB'):
        return False
    if word in {'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                "'m", "'re", "'s"}:
        return False
    # A contraction 've is auxiliary; 'd is normally tagged as a modal.
    if word in {"'ve", "'d"}:
        return False
    if word in {'have', 'has', 'had', 'having', 'do', 'does', 'did'}:
        lookahead = index + 1
        # Adverbs/negation may intervene: has just arrived; does not sing.
        while lookahead < len(tagged) and tagged[lookahead][1].startswith('RB'):
            lookahead += 1
        if lookahead < len(tagged):
            next_word, next_tag = tagged[lookahead]
            if word in {'have', 'has', 'had', 'having'}:
                if next_tag in {'VBN', 'VBG'} or next_word.lower() == 'to':
                    return False
            elif next_tag == 'VB':
                return False
        # Subject-auxiliary inversion: Does the fox know? Has she arrived?
        if tag in {'VB', 'VBP', 'VBZ', 'VBD'}:
            allowed_subject_tags = {'PRP', 'EX', 'DT', 'JJ', 'NN', 'NNS',
                                    'NNP', 'NNPS', 'RB', 'RBR', 'RBS'}
            for j in range(index + 1, min(index + 7, len(tagged))):
                following_tag = tagged[j][1]
                target_tags = {'VBN', 'VBG'} if word in {'have', 'has', 'had', 'having'} else {'VB'}
                if following_tag in target_tags:
                    return False
                if following_tag not in allowed_subject_tags:
                    break
    return True


def calculate_ld(text):
    """English POS-based lexical density: content word tokens / word tokens.

    Nouns, lexical verbs, adjectives and adverbs are counted. Common
    auxiliaries are excluded heuristically; this is an estimated metric,
    not a direct score of readability. Empty/non-word input returns 0.
    """
    if not text.strip() or not re.search(r'[A-Za-z]', text):
        return 0.0
    tagged = _ld_tagged_tokens(text)
    denominator = sum(_is_ld_word(token) for token, _ in tagged)
    if not denominator:
        return 0.0
    numerator = sum(_is_ld_content(tagged, i) for i in range(len(tagged)))
    return numerator / denominator


def word_frequency_analysis(text):
    """计算词频，并移除英文停用词"""
    # 加载NLTK的英文停用词库
    stop_words = set(stopwords.words('english'))
    # 添加自定义的停用词以提高关键词的准确性
    custom_stop_words = {'also', 'would', 'could', 'get', 'like', 'one'}
    stop_words.update(custom_stop_words)

    words = clean_text(text).split()
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    word_counts = Counter(filtered_words)
    return word_counts


def sentiment_analysis(text):
    """对整个文本进行情感分析，返回一个综合情感分数 (compound score)"""
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']


def generate_wordcloud(word_counts, title, output_path):
    """根据词频数据生成并保存词云图片"""
    if not word_counts:
        print(f"警告：因 '{title}' 文本内容为空或无有效词汇，已跳过生成词云。")
        return None

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_counts)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)

    # 清理标题以创建安全的文件名
    safe_title = re.sub('[^a-zA-Z0-9]', '_', title)
    image_path = os.path.join(output_path, f"词云_{safe_title}.png")
    plt.savefig(image_path)
    plt.close()  # 关闭图像以防止在脚本运行时弹出
    return image_path


# --- 数据获取功能模块 ---

def get_text_from_url(url):
    """从指定的URL抓取网页标题和正文文本"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 如果请求失败则引发异常

        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.find('title').get_text().strip() if soup.find('title') else "无标题"

        paragraphs = soup.find_all('p')
        text = '\n'.join([p.get_text() for p in paragraphs])

        return title, text
    except requests.exceptions.RequestException as e:
        print(f"错误：抓取URL {url} 时发生网络错误: {e}")
        return None, None


def get_texts_from_files(paths_str):
    """从用户输入的文件路径字符串中读取所有文件的内容"""
    paths = [path.strip().strip('"') for path in paths_str.split(',')]
    texts_data = []
    for path in paths:
        try:
            title = os.path.basename(path)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            texts_data.append({'title': title, 'text': text, 'source': path})
        except FileNotFoundError:
            print(f"错误：在路径 '{path}' 未找到任何文件。")
        except Exception as e:
            print(f"错误：读取文件 '{path}' 时发生未知错误: {e}")
    return texts_data


def get_texts_from_folder(folder_path):
    """从指定文件夹中读取所有.txt文件的内容"""
    folder_path = folder_path.strip().strip('"')
    texts_data = []
    try:
        if not os.path.isdir(folder_path):
            print(f"错误：提供的路径 '{folder_path}' 不是一个有效的文件夹。")
            return []

        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    texts_data.append({'title': filename, 'text': text, 'source': file_path})
                except Exception as e:
                    print(f"警告：读取文件 '{filename}' 时出错: {e}")

        if not texts_data:
            print(f"提示：在文件夹 '{folder_path}' 中没有找到任何.txt文件。")

    except Exception as e:
        print(f"错误：处理文件夹 '{folder_path}' 时发生未知错误: {e}")

    return texts_data


# --- 报告生成模块 ---

def generate_analysis_report(analysis_results, source_type, source_info):
    """根据分析结果，生成一份完整的Word报告和一份Excel数据表"""
    doc = Document()

    # 设置Word文档标题字体
    try:
        style = doc.styles['Heading 1']
        font = style.font
        font.name = HEADING_FONT_NAME
    except Exception as e:
        print(f"警告：设置字体 '{HEADING_FONT_NAME}' 时出错。请确保该字体已安装。错误信息: {e}")

    # --- 报告主标题 ---
    doc.add_heading('TextBuddy: Analysis Report', level=1)

    doc.add_paragraph(
        "Lexical density is estimated using NLTK English POS tags: nouns, "
        "lexical verbs, adjectives and adverbs divided by word tokens. "
        "Contractions are split; punctuation and numbers are excluded. Common "
        "auxiliaries are excluded heuristically. Tagging may be imperfect; "
        "LD alone is not a readability score. TTR is sensitive to text length."
    )

    # --- 第1部分：TTR 和 词汇密度分析 ---
    doc.add_heading('PART 1: Analysis of Type-token ratio and Lexical Density', level=1)

    # 动态生成每篇文章的介绍
    for i, result in enumerate(analysis_results):
        text_num = i + 1
        title = result['title'] if result['title'] else f"Text {text_num}"
        if source_type == 'url':
            doc.add_paragraph(
                f"The first text picked up from [{source_info}] is a/an {result['type']} titled \"{title}\" (hereinafter referred to as Text {text_num}).")
        else:
            doc.add_paragraph(
                f"The first text is a/an {result['type']} titled \"{title}\" (hereinafter referred to as Text {text_num}).")

    # 根据分析的文章数量，选择不同的结论格式
    if len(analysis_results) == 1:
        # 单篇文章的结论
        result = analysis_results[0]
        doc.add_paragraph(f"\nAccording to the data, the type-token ratio of Text 1 is {result['ttr']:.4f}.")
        doc.add_paragraph(f"In terms of lexical density, the value for Text 1 is {result['ld']:.4f}.")

    elif len(analysis_results) > 1:
        # 多篇文章的比较结论
        highest_ttr_item = max(analysis_results, key=lambda x: x['ttr'])
        highest_ld_item = max(analysis_results, key=lambda x: x['ld'])

        doc.add_paragraph(
            f"\nAccording to the data, Text {analysis_results.index(highest_ttr_item) + 1} has the highest "
            f"type-token ratio ({highest_ttr_item['ttr']:.4f}). As type-token ratio measures lexical diversity, "
            "a higher type-token ratio indicates a higher lexical diversity. It means that more different words "
            "or a greater variety of vocabulary were used and fewer words were repeated in "
            f"Text {analysis_results.index(highest_ttr_item) + 1} than in others."
        )

        ld_text_num = analysis_results.index(highest_ld_item) + 1
        doc.add_paragraph(
            f"\nText {ld_text_num} has the highest estimated lexical density "
            f"({highest_ld_item['ld']:.4f}) among these texts. It contains a "
            "higher proportion of POS-tagged content words under this tool's "
            "definition. This alone does not establish that it is harder to read."
        )
    doc.add_page_break()

    # --- 第2部分：情感分析 ---
    doc.add_heading('PART 2: Sentiment Analysis', level=1)

    # 逐一输出每篇文章的情感分析结果
    for i, result in enumerate(analysis_results):
        score = result['sentiment']
        text_num = i + 1
        if score > 0.05:
            doc.add_paragraph(f"Text {text_num} indicates a positive sentiment (compound score: {score:.4f}).")
        elif score < -0.05:
            doc.add_paragraph(f"Text {text_num} reveals a predominantly negative tone (compound score: {score:.4f}).")
        else:
            doc.add_paragraph(f"Text {text_num} shows a neutral sentiment (compound score: {score:.4f}).")

    doc.add_page_break()

    # --- 第3部分：词频分析 ---
    doc.add_heading('PART 3: Word Frequency Analysis', level=1)

    # 逐一输出每篇文章的词频分析结果和词云
    for i, result in enumerate(analysis_results):
        text_num = i + 1
        word_counts = result['word_freq']
        top_words = word_counts.most_common(3)

        if top_words:
            top_words_str = ", ".join([f"“{word}” (occurs {count} times)" for word, count in top_words])
            doc.add_paragraph(f"\nThe most frequent words in Text {text_num} are {top_words_str}.")

            prominent_words = [word for word, count in word_counts.most_common(4)]
            if len(prominent_words) >= 4:
                doc.add_paragraph(
                    f"The word cloud visualization reveals that “{prominent_words[0]}” is the largest word, "
                    f"representing its high frequency in the text, followed by “{prominent_words[1]}”, "
                    f"“{prominent_words[2]}”, and “{prominent_words[3]}” as other prominent terms."
                )

            if result['wordcloud_path']:
                doc.add_picture(result['wordcloud_path'], width=Inches(6.0))
        else:
            doc.add_paragraph(f"\n文本 {text_num} 在移除停用词后没有足够的内容进行词频分析。")

    # --- 保存Word文档 ---
    doc_path = "TextBuddy_分析报告.docx"
    doc.save(doc_path)
    print(f"\nWord报告已成功保存至: {doc_path}")

    # --- 生成并保存Excel文件 ---
    summary_data = {
        "文本序号": [i + 1 for i in range(len(analysis_results))],
        "标题": [res['title'] for res in analysis_results],
        "来源": [res['source'] for res in analysis_results],
        "类符/形符比 (TTR)": [f"{res['ttr']:.4f}" for res in analysis_results],
        "词汇密度 (LD)": [f"{res['ld']:.4f}" for res in analysis_results],
        "情感综合分数": [f"{res['sentiment']:.4f}" for res in analysis_results],
    }
    summary_df = pd.DataFrame(summary_data)

    excel_path = "TextBuddy_分析报告.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name="分析总览", index=False)
        for i, result in enumerate(analysis_results):
            freq_df = pd.DataFrame(result['word_freq'].most_common(), columns=['词汇', '频率'])
            freq_df.to_excel(writer, sheet_name=f"文本_{i + 1}_词频", index=False)

    print(f"Excel报告已成功保存至: {excel_path}")


# --- 主程序入口 ---

def main():
    # --- 程序启动时的介绍信息 ---
    print("=" * 60)
    print("      欢迎使用 TextBuddy v1.0!")
    print("=" * 60)

    print("\n--- 工具介绍 ---\n")
    print(
        """TextBuddy is your super easy and handy text analysis tool, designed especially for students learning corpus linguistics! Whether it's a news article or a novel, just input the article path or URL , and TextBuddy will instantly calculate Type-Token Ratio (TTR), Lexical Density (LD), Sentiment Analysis, and Word Frequency. The results are then automatically saved in both Word and Excel formats—no more wasting time on tedious calculations! This way, you can focus more on diving into deeper analysis without the hassle of number crunching.""")

    print("\n--- 分析指标说明 ---\n")
    print("""PART 1: Type-token ratio and Lexical Density
In this part, the source texts of different genres will be selected for text analysis in terms of their type-token ratio and lexical density.
Tokens refer to the total count of a word in the text, whereas types represent the count of different words in the text. Type-token ratio (TTR) is the ratio of different words to the number of words in a text or collection of texts. Lexical density (LD) is the ratio of the total number of content words to the total number of tokens. 
The ratio between types and tokens represents lexical diversity, where a higher ratio between types and tokens indicates a greater variety and abundance of words, while lexical density estimates the proportion of content words. LD uses English POS tags: nouns, lexical verbs, adjectives, and adverbs; common auxiliaries are excluded heuristically. Contractions are split into word parts; punctuation and numbers are excluded. LD alone does not determine readability. TTR depends on text length, so compare texts of similar length.""")

    print("\n---\n")
    print("""PART 2: Sentiment Analysis
In this part, the source texts of different genres will be analyzed based on their compound scores, and the analysis results may fall into one of the following four categories: Positive sentiment, Negative sentiment, Neutral sentiment, and Mixed sentiment.
Sentiment Analysis is a fundamental task in Natural Language Processing (NLP), aimed at identifying and classifying the emotions expressed in text. It involves determining whether a piece of text reflects a positive, negative, or neutral sentiment. 
The compound score is a single sentiment score produced by VADER (Valence Aware Dictionary and sEntiment Reasoner) that combines the positive, negative, and neutral scores of a text. It ranges from -1 (most negative) to +1 (most positive), with 0 indicating neutral sentiment. A score greater than 0.05 indicates positive sentiment, less than -0.05 indicates negative sentiment, and values between -0.05 and 0.05 represent neutral sentiment.""")

    print("\n---\n")
    print("""PART 3: Word Frequency Analysis
In this part, the source texts of different genres will be analyzed for word frequency, and an e intuitive word cloud will be generated for each text.
Word Frequency Distribution is a common technique in text analysis that counts how often each word appears in a given text. This helps identify key terms and highlights the most important or frequently mentioned words in the text.
A Word Cloud is a visual representation of the word frequency distribution, where the size of each word is proportional to its frequency. Larger words indicate higher frequency, while smaller words represent lower frequency.""")

    print("\n" + "=" * 60)

    # --- 用户交互部分 ---
    choice = input(
        "\n请选择一个选项:\n1. 分析指定的本地文本文件\n2. 分析网页链接中的文本\n3. 分析指定文件夹中的所有文本文件 (.txt)\n4. 分析内置短篇童话 The Fox and the Cat\n请输入您的选择 (1, 2, 3 或 4): ")

    texts_to_process = []
    source_type = ""
    source_info = ""

    if choice == '1':
        source_type = "file"
        paths_input = input(
            "\n请输入一个或多个文件的完整路径，用英文逗号分隔:\n(例如: C:\\Users\\YourName\\doc1.txt, D:\\data\\story.txt)\n> ")
        texts_to_process = get_texts_from_files(paths_input)
        source_info = "本地文件"

    elif choice == '2':
        source_type = "url"
        url = input("\n请输入需要分析的网页链接: ")
        title, text = get_text_from_url(url)
        if text:
            texts_to_process.append({'title': title, 'text': text, 'source': url})
        source_info = url

    elif choice == '3':
        source_type = "folder"
        folder_path = input("\n请输入包含.txt文件的文件夹的完整路径:\n(例如: C:\\Users\\YourName\\MyCorpus)\n> ")
        texts_to_process = get_texts_from_folder(folder_path)
        source_info = folder_path

    elif choice == '4':
        source_type = "sample"
        source_info = SAMPLE_SOURCE
        texts_to_process.append({'title': SAMPLE_TITLE, 'text': SAMPLE_TEXT,
                                 'source': SAMPLE_SOURCE})
        print("\nSample: " + SAMPLE_TITLE + " — Brothers Grimm")
        print("English text based on translations by Edgar Taylor and Marian Edwardes.")
        print("Source: " + SAMPLE_SOURCE)

    else:
        print("无效选择，程序即将退出。")
        return

    # 如果没有加载到任何文本，则退出
    if not texts_to_process:
        print("未能加载任何文本进行分析，程序即将退出。")
        return

    # 为每个加载的文本获取文章类型
    for data in texts_to_process:
        article_type = input(f"请输入 '{data['title']}' 的文章类型 (例如: 新闻, 小说, 学术): ")
        data['type'] = article_type

    # --- 对所有待处理文本执行分析 ---
    analysis_results = []
    output_dir = "TextBuddy_输出文件"
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- 开始分析 ---")
    for i, data in enumerate(texts_to_process):
        print(f"正在分析文本 {i + 1}: '{data['title']}'...")
        text = data['text']
        ttr = calculate_ttr(text)
        ld = calculate_ld(text)
        sentiment = sentiment_analysis(text)
        word_counts = word_frequency_analysis(text)
        wordcloud_path = generate_wordcloud(word_counts, data['title'], output_dir)

        analysis_results.append({
            'title': data['title'], 'source': data['source'], 'type': data['type'],
            'ttr': ttr, 'ld': ld, 'sentiment': sentiment, 'word_freq': word_counts,
            'wordcloud_path': wordcloud_path
        })
    print("--- 分析完成 ---")

    # --- 生成最终报告 ---
    if analysis_results:
        print("\n正在生成Word和Excel报告...")
        generate_analysis_report(analysis_results, source_type, source_info)
        print("\n所有任务已成功完成！")
    else:
        print("没有可供分析的文本，因此未生成任何报告。")


# 当脚本被直接执行时，运行main函数
if __name__ == "__main__":
    main()

