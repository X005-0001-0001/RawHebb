# -*- coding: utf-8 -*-
import io

p = r'D:\explore\wav\_v163_shell19.py'
t = io.open(p, encoding='utf-8').read()

old_start = t.index('def truncate_messages(msgs, tok, max_len):')
old_end = t.index('class RawHebbTokenizer', old_start)
NEW = '''def _render_tool_calls(tcs):
    return __import__('json').dumps(tcs, ensure_ascii=False) if tcs else ''


def _assistant_segments(msg):
    """与 3120012 的 render_messages 一致：reasoning_content + content + tool_calls。"""
    segs = []
    rc = msg.get('reasoning_content')
    if rc:
        segs.append(str(rc))
    c = msg.get('content')
    segs.append('' if c is None else str(c))
    tc = _render_tool_calls(msg.get('tool_calls'))
    if tc:
        segs.append(tc)
    return ''.join(segs)


def message_token_parts(tok, msg):
    """把一条消息渲染成 (ids, train_flags)。格式与 3120012 的 render_messages 对齐：
    system → <|user|>[系统]…<|end|>；tool → <|user|>[工具结果…]…<|end|>；
    assistant → <|assistant|>…<|end|>，其中内容与 <|end|> 参与训练。"""
    role = msg.get('role')
    content = '' if msg.get('content') is None else str(msg.get('content'))
    if role == 'system':
        return [tok.user_token_id] + tok.encode('[系统]' + content) + [tok.end_token_id], False
    if role == 'user':
        return [tok.user_token_id] + tok.encode(content) + [tok.end_token_id], False
    if role == 'tool':
        tid = msg.get('tool_call_id')
        head = '[工具结果' + ((' id=' + str(tid)) if tid else '') + ']'
        return [tok.user_token_id] + tok.encode(head + content) + [tok.end_token_id], False
    if role == 'assistant':
        cids = tok.encode(_assistant_segments(msg))
        return [tok.assistant_token_id] + cids + [tok.end_token_id], True
    return [], False


def _message_cost(tok, msg):
    ids, _ = message_token_parts(tok, msg)
    return len(ids)


def truncate_messages(msgs, tok, max_len):
    """3120012 的 _truncate_pairs：保留开头连续 system + 最近的完整 (user, assistant) 对。
    预算是真实 token 数（含 role/end 与 reasoning/tool_calls），不是字符数。"""
    head, start = [], 0
    while start < len(msgs) and msgs[start].get('role') == 'system':
        head.append(msgs[start])
        start += 1
    budget = sum(_message_cost(tok, m) for m in head) + 2       # + bos/eos
    kept, i = [], len(msgs) - 1
    while i >= start + 1:
        if msgs[i].get('role') == 'assistant' and msgs[i - 1].get('role') == 'user':
            pair = [msgs[i - 1], msgs[i]]
            budget += sum(_message_cost(tok, m) for m in pair)
            kept.insert(0, pair)
            if budget + 2 >= max_len:                            # 与 312 相同：先收下最后一对
                break
            i -= 2
        else:
            i -= 1
    flat = [m for p in kept for m in p]
    if not flat:                                                 # 非标准对话：退回最近整条消息
        for m in reversed(msgs[start:]):
            c = _message_cost(tok, m)
            if flat and budget + c > max_len:
                break
            flat.insert(0, m)
            budget += c
    return head + flat


'''
t = t[:old_start] + NEW + t[old_end:]
io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
print('truncate_messages replaced')
