# AC 自动机与保护性词典协同重构设计文档（详细接口与伪代码）

> 时间戳：2024-07-04

## 一、目标与原则

- **批量保护**：对所有白名单短语（如专业术语、专有名词等）进行统一保护，防止被分词器拆分。
- **高效统计**：BM25 词典统计时，能正确统计所有白名单短语的出现频次。
- **接口清晰**：各功能模块职责单一，便于单元测试和后续维护。
- **可扩展性**：支持后续扩展更多短语保护/还原策略。

---

## 二、核心接口设计

### 1. PhraseMasker（短语掩码器）

#### 1.1 构造函数

```python
class PhraseMasker:
    def __init__(self, phrases: List[str]):
        # phrases: 白名单短语列表
        # 初始化AC自动机
        # 生成 phrase <-> placeholder 映射
```

#### 1.2 批量掩码接口

```python
def mask_texts(self, texts: List[str], embed_phrase: bool = True) -> Tuple[List[str], List[List[Tuple]], Dict[str, str]]:
    """
    对一批文本做短语掩码。
    :param texts: 原始文本列表
    :param embed_phrase: 是否在占位符中嵌入原始短语
    :return: (masked_texts, intervals_list, phrase2placeholder)
        - masked_texts: 掩码后的文本列表
        - intervals_list: 每条文本的短语区间及占位符信息
        - phrase2placeholder: phrase -> placeholder 映射
    """
```

#### 1.3 占位符还原接口

```python
def restore_text(self, masked_text: str, intervals: List[Tuple]) -> str:
    """
    将掩码文本还原为原始文本。
    :param masked_text: 掩码后的文本
    :param intervals: 掩码时记录的区间信息
    :return: 还原后的文本
    """
```

---

### 2. 分词与统计主流程

#### 2.1 分词接口

```python
def tokenize_texts(texts: List[str], userdict_path: str) -> List[List[str]]:
    """
    用jieba+userdict对一批文本分词。
    :param texts: 文本列表
    :param userdict_path: 保护性词典路径
    :return: 分词结果列表
    """
```

#### 2.2 BM25词典统计接口

```python
def build_bm25_vocab(token_lists: List[List[str]], phrase2placeholder: Dict[str, str]) -> Dict[str, int]:
    """
    统计BM25词典，自动将占位符还原为原始短语。
    :param token_lists: 分词结果列表
    :param phrase2placeholder: phrase -> placeholder 映射
    :return: 词频字典 {词: 频次}
    """
```

---

### 3. 主流程接口

```python
def process_texts_with_ac_mask(
    texts: List[str],
    phrases: List[str],
    userdict_path: str,
    embed_phrase: bool = True
) -> Dict[str, int]:
    """
    主流程：批量掩码、分词、统计BM25词典。
    :param texts: 原始文本列表
    :param phrases: 白名单短语列表
    :param userdict_path: 保护性词典路径
    :param embed_phrase: 是否在占位符中嵌入原始短语
    :return: BM25词频字典
    """
    # 1. 批量掩码
    masker = PhraseMasker(phrases)
    masked_texts, intervals_list, phrase2placeholder = masker.mask_texts(texts, embed_phrase=embed_phrase)
    # 2. 分词
    token_lists = tokenize_texts(masked_texts, userdict_path)
    # 3. 统计
    vocab = build_bm25_vocab(token_lists, phrase2placeholder)
    return vocab
```

---

## 三、伪代码实现

```python
# 1. 批量掩码
masker = PhraseMasker(phrases)
masked_texts, intervals_list, phrase2placeholder = masker.mask_texts(texts, embed_phrase=True)

# 2. 分词
import jieba
jieba.load_userdict(userdict_path)
token_lists = [list(jieba.cut(text)) for text in masked_texts]

# 3. 统计
vocab = {}
for tokens in token_lists:
    for token in tokens:
        # 判断是否为占位符
        if token.startswith("__PHRASE_"):
            # 方案一：占位符中直接嵌入原始短语
            # 例如 __PHRASE_0_阿苏焉尼__，直接提取"阿苏焉尼"
            phrase = token.split("_", 2)[-1].rstrip("_")
            vocab[phrase] = vocab.get(phrase, 0) + 1
        else:
            vocab[token] = vocab.get(token, 0) + 1
```

---

## 四、参数说明

- **texts**：原始文本列表，支持批量处理。
- **phrases**：白名单短语列表，支持任意数量。
- **userdict_path**：保护性词典路径，保证分词时短语不被拆分。
- **embed_phrase**：是否在占位符中嵌入原始短语，推荐为 True。

---

## 五、扩展与测试建议

- **PhraseMasker** 可扩展支持不同占位符格式、不同还原策略。
- **tokenize_texts** 可支持多种分词器（如jieba、thulac等）。
- **build_bm25_vocab** 可扩展为支持更多统计特征（如DF、PMI等）。
- **单元测试**：建议对掩码、分词、还原、统计各环节分别做测试，确保每个短语都能被保护和统计。

---

## 六、总结

- 该设计适用于**所有白名单短语**的批量保护和统计。
- 各接口职责清晰，便于维护和扩展。
- 方案兼容 AC 自动机和 jieba userdict 的优点，能高效、准确地保护和统计所有专业短语。

---

如需进一步细化某个接口的实现细节或示例代码，请随时告知！ 