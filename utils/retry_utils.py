"""
retry_utils.py
==============

提供统一的网络重试装饰器，用于处理各种API调用的超时和重试逻辑。
支持通用化和定制化的重试策略。
"""

import time
import logging
import functools
from typing import Callable, Any, Optional
from config import (
    EMBEDDING_CONNECT_TIMEOUT, EMBEDDING_READ_TIMEOUT,
    EMBEDDING_MAX_RETRIES, EMBEDDING_RETRY_DELAY,
    EMBEDDING_USE_EXPONENTIAL_BACKOFF, EMBEDDING_BACKOFF_FACTOR,
    EMBEDDING_MAX_BACKOFF_DELAY
)

def api_retry_decorator(service_name: str = "API",
                        max_retries: int = None, 
                        retry_delay: float = None,
                        use_backoff: bool = None,
                        retry_on_exceptions: tuple = (Exception,)):
    """
    通用API重试装饰器
    
    Args:
        service_name: 服务名称，用于日志记录
        max_retries: 最大重试次数，默认使用配置值
        retry_delay: 重试延迟（秒），默认使用配置值
        use_backoff: 是否使用指数退避，默认使用配置值
        retry_on_exceptions: 需要重试的异常类型，默认所有异常
    
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            max_retries_actual = max_retries or EMBEDDING_MAX_RETRIES
            retry_delay_actual = retry_delay or EMBEDDING_RETRY_DELAY
            use_backoff_actual = use_backoff if use_backoff is not None else EMBEDDING_USE_EXPONENTIAL_BACKOFF
            
            last_exception = None
            
            for attempt in range(max_retries_actual + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on_exceptions as e:
                    last_exception = e
                    if attempt < max_retries_actual:
                        # 计算延迟时间
                        if use_backoff_actual:
                            delay = min(retry_delay_actual * (EMBEDDING_BACKOFF_FACTOR ** attempt), 
                                      EMBEDDING_MAX_BACKOFF_DELAY)
                        else:
                            delay = retry_delay_actual
                        
                        logging.warning(f"{service_name}调用失败 (尝试 {attempt + 1}/{max_retries_actual + 1}): {e}")
                        logging.info(f"等待 {delay:.1f} 秒后重试...")
                        time.sleep(delay)
                    else:
                        logging.error(f"{service_name}调用最终失败，已重试 {max_retries_actual} 次: {e}")
                        raise last_exception
            
            return None  # 理论上不会执行到这里
        return wrapper
    return decorator

def embedding_retry_decorator(max_retries: int = None, 
                             retry_delay: float = None,
                             use_backoff: bool = None):
    """
    Embedding专用重试装饰器（向后兼容）
    
    Args:
        max_retries: 最大重试次数，默认使用配置值
        retry_delay: 重试延迟（秒），默认使用配置值
        use_backoff: 是否使用指数退避，默认使用配置值
    
    Returns:
        装饰后的函数
    """
    return api_retry_decorator(
        service_name="Embedding",
        max_retries=max_retries,
        retry_delay=retry_delay,
        use_backoff=use_backoff
    )

def network_retry_decorator(service_name: str = "Network",
                           max_retries: int = 3,
                           retry_delay: float = 1.0,
                           use_backoff: bool = True,
                           retry_on_exceptions: tuple = (Exception,)):
    """
    网络请求专用重试装饰器
    
    Args:
        service_name: 服务名称
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        use_backoff: 是否使用指数退避
        retry_on_exceptions: 需要重试的异常类型
    
    Returns:
        装饰后的函数
    """
    return api_retry_decorator(
        service_name=service_name,
        max_retries=max_retries,
        retry_delay=retry_delay,
        use_backoff=use_backoff,
        retry_on_exceptions=retry_on_exceptions
    )

def database_retry_decorator(service_name: str = "Database",
                            max_retries: int = 3,
                            retry_delay: float = 0.5,
                            use_backoff: bool = True,
                            retry_on_exceptions: tuple = (Exception,)):
    """
    数据库操作专用重试装饰器
    
    Args:
        service_name: 服务名称
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        use_backoff: 是否使用指数退避
        retry_on_exceptions: 需要重试的异常类型
    
    Returns:
        装饰后的函数
    """
    return api_retry_decorator(
        service_name=service_name,
        max_retries=max_retries,
        retry_delay=retry_delay,
        use_backoff=use_backoff,
        retry_on_exceptions=retry_on_exceptions
    )
