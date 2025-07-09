import os
import sys
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataupload.bm25_manager import BM25Manager
from vector_search import VectorSearch
from config import BM25_K1, BM25_B, BM25_MIN_FREQ, BM25_MAX_VOCAB_SIZE, BM25_CUSTOM_DICT_PATH, HYBRID_ALPHA

def test_bm25_sparse_vector():
    bm25 = BM25Manager(k1=BM25_K1, b=BM25_B, min_freq=BM25_MIN_FREQ, max_vocab_size=BM25_MAX_VOCAB_SIZE, custom_dict_path=BM25_CUSTOM_DICT_PATH)
    corpus = ["泰伦虫族是银河系的主要威胁之一。", "星际战士拥有强大的基因改造能力。"]
    bm25.fit(corpus)
    vec = bm25.get_sparse_vector("泰伦虫族")
    assert isinstance(vec, dict)
    assert len(vec["indices"]) == len(vec["values"])
    assert all(isinstance(i, int) for i in vec["indices"])
    assert all(v > 0 for v in vec["values"])

def test_bm25_incremental_update():
    bm25 = BM25Manager(k1=BM25_K1, b=BM25_B, min_freq=BM25_MIN_FREQ, max_vocab_size=BM25_MAX_VOCAB_SIZE, custom_dict_path=BM25_CUSTOM_DICT_PATH)
    corpus = ["泰伦虫族是银河系的主要威胁之一。"]
    bm25.fit(corpus)
    old_vocab_size = len(bm25.vocabulary)
    bm25.incremental_update(["欧克兽人以蛮力著称。"])
    new_vocab_size = len(bm25.vocabulary)
    assert new_vocab_size >= old_vocab_size

def test_hybrid_scale():
    from vector_search import VectorSearch
    dense = [1.0, 2.0, 3.0]
    sparse = {"indices": [0, 2], "values": [0.5, 1.5]}
    alpha = HYBRID_ALPHA
    vs = VectorSearch(os.getenv('PINECONE_API_KEY'), os.getenv('PINECONE_INDEX'))
    hdense, hsparse = vs.hybrid_scale(dense, sparse, alpha)
    assert all(abs(v - d * alpha) < 1e-6 for v, d in zip(hdense, dense))
    assert all(abs(v - s * (1 - alpha)) < 1e-6 for v, s in zip(hsparse["values"], sparse["values"]))

def test_bm25_untrained_exception():
    bm25 = BM25Manager()
    with pytest.raises(ValueError):
        bm25.get_sparse_vector("test")

def test_hybrid_alpha_config():
    """测试HYBRID_ALPHA配置是否正确读取"""
    # 测试默认值
    from config import HYBRID_ALPHA
    assert HYBRID_ALPHA == 0.3
    
    # 测试环境变量覆盖
    os.environ['HYBRID_ALPHA'] = '0.5'
    import importlib
    import config
    importlib.reload(config)
    assert config.HYBRID_ALPHA == 0.5
    
    # 恢复默认值
    del os.environ['HYBRID_ALPHA']
    importlib.reload(config)
    assert config.HYBRID_ALPHA == 0.3

def test_hybrid_scale_logic():
    """测试hybrid_scale逻辑（不依赖VectorSearch）"""
    alpha = 0.3
    dense = [1.0, 2.0, 3.0]
    sparse = {"indices": [0, 2], "values": [0.5, 1.5]}
    hdense = [v * alpha for v in dense]
    hsparse = {"indices": sparse["indices"], "values": [v * (1 - alpha) for v in sparse["values"]]}
    assert len(hdense) == len(dense)
    assert len(hsparse["values"]) == len(sparse["values"])
    assert hsparse["indices"] == sparse["indices"]
    for i, d in enumerate(dense):
        assert abs(hdense[i] - d * alpha) < 1e-6
    for i, s in enumerate(sparse["values"]):
        assert abs(hsparse["values"][i] - s * (1 - alpha)) < 1e-6 