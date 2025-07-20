from openai import OpenAI
from pathlib import Path
from config import OPENAI_API_KEY, LLM_MODEL

# 使用新版本的 OpenAI 客户端
client = OpenAI(api_key=OPENAI_API_KEY)


def call_gpt(prompt):
    """使用新版本的 OpenAI API 调用 GPT"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"调用 GPT 时发生错误: {e}")
        return None

# 读取你提供的 txt 句子列表
sentences = Path("1.txt").read_text(encoding="utf-8").splitlines()
sentences = [s.strip() for s in sentences if s.strip()]

outputs = []
for i, sentence in enumerate(sentences):
    prompt = f"""
你是一个中文自然语言理解系统，需要对输入句子进行 slot 标注任务，使用 BIO 格式（每行一个 token 和标签），用制表符隔开。请先对输入句子进行合理的中文分词，然后对每个词进行 slot 标注
这是输入的句子:{sentence}
请使用以下 slot 标签体系：
- B-UNIT / I-UNIT：具体单位
- B-UNIT_TYPE / I-UNIT_TYPE：军队单位还是模型
- B-CONDITION / I-CONDITION：判定的条件
- B-ACTION / I-ACTION：执行的动作
- B-SKILL/I-SKILL: 具体的技能名称
- B-SKILL_TYPE/I_SKILL_TYPE: 技能类型(是关键字, 技能,还是战略计谋)
- B-WHEN: 当..时
- B-PASSIVE: 表示被动接收结果
- B-ATTACK_TYPE/I-ATTACK_TYPE: 攻击的类型
- B-THRESHOLD/I-THRESHOLD: 超过或低于某个阈值
- B-RESULT/I-RESULT: 具体的结果或内容
- B-BOOL: 表达是否
- B-WEAPON: 泛指武器或者具体的武器名称
- B-ATTR/I-ATTR: 具体的某个属性值
- B-WHICH: 哪一个,代表要选出具体的一个结果
- B-HOWMANY: 多少个,哪些,一个范围内所有的结果
- B-COMPARE/I-COMPARE:  采纳哪一类的比较结果, 更高/低, or 更大/小..
- B-NUMBER: 用来代指数量或具体的数量
- B-PHASE: 战斗阶段
- B-TERM: 战锤 40k 里的专有词汇
- B-WEAPON_KWS: 战锤 40K 里武器的关键字
- B-COMPARE_WITH: 用于两个对象比较之间的连接词
- O：非实体词

请按照如下格式输出：
【每个字/词】<tab>【标签】
句子之间用空行隔开。不需要输出任何解释说明。

### 示例：
输入句子：拥有"独行特攻"技能的单位(例如独角)在什么条件下可以被远程攻击命中?
输出：

拥有        O
"           O
独行特工      B-SKILL
"           O
技能        B-SKILL_TYPE
的          O
单位        B-UNIT_TYPE
（          O
例如        O
独角        B-UNIT
）          O
在          B-WHEN
什么        B-CONDITION
条件        I-CONDITION
下          O
可以        B-BOOL
被          B-PASSIVE
远程        B-ATTACK_TYPE
攻击        I-ATTACK_TYPE
选中        I-ACTION
？          O

    """
    result = call_gpt(prompt)
    if result:
        outputs.append(result.strip())
        print(f"[✓] 第{i+1}句完成")
    else:
        print(f"[✗] 第{i+1}句处理失败")

# 保存结果
Path("slot_filling_output.txt").write_text("\n\n".join(outputs), encoding="utf-8")
print(f"处理完成，共处理 {len(outputs)} 句，结果已保存到 slot_filling_output.txt")
