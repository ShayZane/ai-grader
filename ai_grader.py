# -*- coding: utf-8 -*-
"""
AI 判分核心逻辑
------------------------------------------------------------------
把「题目 + 评分标准 + 学生答案」拼成一段提示词，交给大模型，
拿回一个 JSON：总分、逐条得分、评语。

对外只暴露一个函数：grade_answer(...)
  - 有 DEEPSEEK_API_KEY  → 调用真实大模型
  - 没有 Key             → 走「模拟打分」，保证流程照样能演示

注意：Key 只从环境变量读取，绝不写死在代码里。
"""

import os
import re
import json

# 自动读取项目目录下的 .env 文件（Key 就放在那里）
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- 提示词：决定 AI 打得准不准，是项目的灵魂 ----------------------

SYSTEM_PROMPT = """你是一位严格、较真的阅卷老师。你只认评分标准上白纸黑字写的要求，不给人情分。

评分方法（必须严格执行）：
1. 逐条看评分标准。对每一条，先判断学生答案属于哪一档：
   - 完全达到 → 这一条给满分
   - 部分达到 → 这一条给一半的分
   - 没有做到 → 这一条给 0 分
2. 只要这一条里有任何一个要求没做到，就不许给满分。
3. 如果评分标准涉及代码的实际行为（变量相加、字符串拼接、语法错误等），
   你要像在脑子里运行一遍那样推演它真正的输出，再判断对不对。
   例如：input() 拿到的是文字，"30" + "15" 的结果是 "3015"，不是 45。
4. 每一条的 comment 必须引用学生答案里的原文片段作为依据，不能凭感觉说。
5. 不要因为答案"整体看起来还行"就放宽某一条。宁可严，不可松。
6. 不要给某一条附加评分标准里没写的额外要求。每一条只看它字面上写了什么。
7. 总分 = 各条得分之和，不能超过满分。
8. 只输出一个 JSON 对象，不要输出任何其他文字、不要用 markdown 代码块。"""

USER_TEMPLATE = """【题目】
{question}

【评分标准】
{criteria}

【学生答案】
{answer}

请严格按下面的 JSON 格式输出：
{{
  "score": 总分(数字),
  "max_score": 满分(数字),
  "details": [
    {{"item": "评分点原文", "level": "达到 或 部分达到 或 未达到", "point": 该条得分, "comment": "引用学生答案原文说明"}}
  ],
  "reason": "总体评语"
}}"""


# ---- 对外主函数 ----------------------------------------------------

def grade_answer(question, criteria, answer, max_score=100, temperature=0.0):
    """
    参数：题目、评分标准、学生答案、满分
    返回：(结果字典, 模式字符串)   模式为 "deepseek" 或 "mock"
    """
    api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()

    if not api_key or api_key.startswith("sk-你的"):
        return _mock_grade(question, criteria, answer, max_score), "mock"

    # ---- 真实调用（DeepSeek 兼容 OpenAI 接口，所以用 openai 这个包）----
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            question=question, criteria=criteria, answer=answer)},
    ]
    preferred = (os.getenv("DEEPSEEK_MODEL") or "deepseek-chat").strip()
    text, used_model = _chat_with_fallback(client, messages, temperature, preferred)

    data = _normalize(_parse_json(text), max_score)
    data["model"] = used_model
    return data, "deepseek"


# ---- 内部工具函数 --------------------------------------------------

# 模型名字可能随平台更新变化，这里按顺序试，谁通用谁
_MODEL_CANDIDATES = ("deepseek-chat", "deepseek-flash", "deepseek-pro")


def _chat_with_fallback(client, messages, temperature, preferred):
    """先试配置里的模型；如果平台说这个模型不存在，就换下一个候选。"""
    tried, last_err = [], None
    order = [preferred] + [m for m in _MODEL_CANDIDATES if m != preferred]

    for model in order:
        tried.append(model)
        try:
            resp = _create(client, model, messages, temperature)
            return resp.choices[0].message.content, model
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            # 只有「模型不存在」这类错误才换模型，别的错误（如 Key 错）直接抛出去
            if "model" in msg and any(k in msg for k in
                                     ("not exist", "not found", "invalid", "unknown", "does not")):
                continue
            raise

    raise RuntimeError("试过这些模型都不可用：%s；最后一个错误：%s" % (tried, last_err))


def _create(client, model, messages, temperature):
    """有的模型不支持 response_format，就退回普通调用。"""
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},  # 让模型直接吐 JSON，好解析
        )
    except Exception as e:
        msg = str(e).lower()
        if "response_format" in msg or "json" in msg:
            return client.chat.completions.create(
                model=model, messages=messages, temperature=temperature)
        raise


def _parse_json(text):
    """模型偶尔会多写几个字，这里把 JSON 抠出来。"""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"score": 0, "reason": "模型返回格式无法解析，请重试", "details": []}


def _normalize(data, max_score):
    """
    保证字段齐全、分数不越界。

    重要：总分不采用模型自己报的 score，而是把 details 里每条的 point 加总。
    实测模型经常"逐条打 20+20+15+0，却把总分写成 70"——它不会算数，
    所以总分交给程序算，AI 只负责逐条判分。
    """
    details = data.get("details", []) or []
    out = {
        "max_score": max_score,
        "details": details,
        "reason": data.get("reason", ""),
    }

    total, has_point = 0.0, False
    for d in details:
        try:
            total += float(d.get("point", 0))
            has_point = True
        except (TypeError, ValueError):
            continue

    if has_point:
        out["score"] = total
    else:
        # 没有逐条明细时，才退回用模型报的总分
        try:
            out["score"] = float(data.get("score", 0))
        except (TypeError, ValueError):
            out["score"] = 0.0

    out["score"] = max(0.0, min(float(max_score), out["score"]))
    out["score"] = round(out["score"], 1)
    if out["score"] == int(out["score"]):
        out["score"] = int(out["score"])
    return out


def _mock_grade(question, criteria, answer, max_score):
    """
    模拟打分：没配 API Key 时用，保证流程能跑通、能演示。
    做法很朴素——按「评分标准的每一行」去学生答案里找覆盖情况，
    覆盖得越多分越高。只是占位，真实打分请配置 DEEPSEEK_API_KEY。
    """
    lines = [l.strip() for l in (criteria or "").splitlines() if l.strip()]
    if not lines:
        lines = ["整体完成度（100分）"]

    # 每行的分数权重：优先用行末括号里写的分数，没写就平均分
    weights = []
    for l in lines:
        m = re.search(r"[（(]\s*(\d+(?:\.\d+)?)\s*分?\s*[)）]", l)
        weights.append(float(m.group(1)) if m else 0.0)
    if sum(weights) <= 0:
        weights = [1.0] * len(lines)

    ans = answer or ""
    total_w = sum(weights)
    details, total = [], 0.0

    for line, w in zip(lines, weights):
        need = re.sub(r"[（(][^）)]*[)）]", "", line)          # 去掉行末的（XX分）
        need = re.sub(r"^\s*\d+\s*[.、)]\s*", "", need)         # 去掉开头的序号
        need_clean = re.sub(r"\s+", "", need)
        grams = [need_clean[i:i + 2] for i in range(len(need_clean) - 1)] or [need_clean]
        grams = [g for g in grams if g.strip()]
        hit = sum(1 for g in grams if g in ans)
        ratio = hit / len(grams) if grams else 0.0
        # 开方做一点放大，让「覆盖大半」的答案不至于被压到很低的分
        point = round(max_score * w / total_w * (ratio ** 0.5), 1)
        total += point
        details.append({
            "item": line,
            "level": "达到" if ratio >= 0.7 else ("部分达到" if ratio >= 0.3 else "未达到"),
            "point": point,
            "comment": "对应要点覆盖约 %d%%" % int(ratio * 100),
        })

    total = round(total, 1)
    if total == int(total):
        total = int(total)
    return {
        "score": total,
        "max_score": max_score,
        "details": details,
        "reason": "【模拟打分】按评分标准的要点覆盖情况估算，仅供演示流程用。"
                  "配置 DEEPSEEK_API_KEY 后会切换为真实大模型阅卷。",
    }
