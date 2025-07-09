import os
import sys
import re
import jieba
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataupload.bm25_manager import BM25Manager

files = [
    'test/testdata/aeldaricodex.md',
]

corpus = []
for f in files:
    with open(f, 'r', encoding='utf-8') as fin:
        corpus.append(fin.read())

bm25 = BM25Manager()
bm25.fit(corpus)
stats = bm25.get_vocabulary_stats()

# 标点符号
punctuations = set('，。！？：；、"''()[]{}<>|/~`+=-_')
# 人称代词
pronouns = set(['你', '我', '他', '她', '它', '我们', '你们', '他们', '她们', '它们', '本人', '自己', '咱们'])
# 常见助词（可扩展）
particles = set(['的', '了', '着', '过', '和', '与', '及', '或', '也', '都', '就', '还', '被', '把', '在', '是', '为', '对', '以', '于', '而', '并', '从', '到', '由', '向', '给', '被', '让', '被', '把', '将', '呢', '吗', '吧', '啊', '呀', '嘛', '哦', '呗', '啦', '哇', '么', '嘛', '喽', '罢', '咧', '哟', '咦', '呦', '哎', '欸', '呃', '呜', '呗', '呸', '哼', '唉', '嘻', '嘶', '嘭', '噢', '噗', '噼', '哗', '咚', '咔', '咯', '咳', '咩', '咪', '咻', '咿', '咽', '咱', '咬', '咳', '咦', '咧', '咱', '咩', '咻', '咿', '咽', '咬', '咳', '咦', '咧'])

filtered = []
for word, idx in bm25.vocabulary.items():
    if word in punctuations or word in pronouns or word in particles:
        continue
    if len(word) == 1 and word not in set('战术规则军队单位模型分队技能'):  # 单字一般无信息量，保留少量领域高频字
        continue
    if re.match(r'^#+$', word):  # 多个#
        continue
    if re.match(r'^\s+$', word):  # 全空白
        continue
    if re.match(r'^<.*>$', word):  # html标签
        continue
    filtered.append((word, idx))

print(f"BM25词典原始大小: {stats['vocab_size']}")
print(f"过滤后有效内容词数量: {len(filtered)}")
print(f"前50内容词(过滤后): {filtered[:50]}") 