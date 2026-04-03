"""UAD训练工具函数"""

def format_training_query(query: str, input_text: str) -> str:
    """
    格式化训练查询
    
    Args:
        query: instruction文本
        input_text: 输入数据
    
    Returns:
        格式化后的prompt
    """
    if input_text:
        return f"{query}\n{input_text}"
    return query
