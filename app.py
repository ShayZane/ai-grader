# -*- coding: utf-8 -*-
"""
AI 判分系统 · 主界面
------------------------------------------------------------------
运行：streamlit run app.py     （然后用浏览器打开它给的本地地址）

页面结构：
  ① 老师填「题目 + 评分标准」
  ② 学生提交「答案」
  ③ AI 给出「分数 + 逐条明细 + 理由」
  ④ 老师改分 → 确认 → 记录进表格，可导出 CSV 做对比测试
"""

import os
import json
import time
import datetime as dt

import streamlit as st

from ai_grader import grade_answer

# ---------------- 页面基础设置 ----------------
st.set_page_config(page_title="AI 判分系统", page_icon="📝", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 读入示例题库（演示时能一键载入）
with open(os.path.join(BASE_DIR, "sample_cases.json"), "r", encoding="utf-8") as f:
    SAMPLES = json.load(f)
CASE_NAMES = list(SAMPLES["cases"].keys())

# ---------------- 侧边栏：当前运行状态 ----------------
with st.sidebar:
    st.header("运行状态")
    if (os.getenv("DEEPSEEK_API_KEY") or "").strip().startswith("sk-") \
            and not os.getenv("DEEPSEEK_API_KEY").startswith("sk-你的"):
        st.success("已检测到 API Key：AI 真实打分中")
        mode_hint = "真实大模型"
    else:
        st.warning("未检测到 API Key：当前是【模拟打分】模式\n\n"
                   "在项目目录新建 .env 文件，写入 DEEPSEEK_API_KEY=你的key，"
                   "重启程序即可切换为真实打分。")
        mode_hint = "模拟打分"
    st.caption(f"当前模式：{mode_hint}")

    st.markdown("---")
    st.markdown("**3 分钟演示三步走**")
    st.markdown(
        "1. 选一道示例题 → 载入示例答案 A → 点「AI 打分」\n"
        "2. 换成示例答案 B，再点一次 → 分数和理由跟着变\n"
        "3. 老师改分 → 确认 → 下方表格出现记录"
    )
    st.markdown("---")
    st.caption("Key 只从 .env 读取，不写进代码，也不会上传。")

# ---------------- 标题 ----------------
st.title("AI 作业批改助手")
st.caption("老师出题 + 写评分标准 → 学生提交答案 → AI 给出分数和理由 → 老师改分确认")

# ---------------- 左：题目与评分标准 / 右：学生答案 ----------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("① 题目与评分标准")
    picked = st.selectbox("载入示例题目（也可以全部手动输入）", ["（手动输入）"] + CASE_NAMES)

    default_q, default_c, sample_answers = "", "", []
    if picked != "（手动输入）":
        case = SAMPLES["cases"][picked]
        default_q, default_c = case["question"], case["criteria"]
        sample_answers = case.get("answers", [])

    # key 带上 picked，切换示例时输入框内容会跟着换
    question = st.text_area("题目", value=default_q, height=120, key=f"q_{picked}")
    criteria = st.text_area("评分标准（一行一条，写上每条的分数）",
                            value=default_c, height=220, key=f"c_{picked}")
    max_score = st.number_input("满分", min_value=1, max_value=200, value=100, step=5)

with col_right:
    st.subheader("② 学生答案")
    opts = ["（手动输入）"] + [f"示例答案 {a['label']}" for a in sample_answers]
    pick_a = st.selectbox("载入示例答案（演示时换这个，看分数变化）", opts)
    if pick_a != "（手动输入）":
        idx = opts.index(pick_a) - 1
        answer_default = sample_answers[idx]["text"]
    else:
        answer_default = ""

    student_name = st.text_input("学生姓名 / 学号", value="同学A")
    answer = st.text_area("答案内容", value=answer_default, height=258, key=f"a_{picked}_{pick_a}")

# ---------------- 打分按钮 ----------------
if st.button("AI 打分", type="primary", use_container_width=True):
    if not question.strip() or not criteria.strip() or not answer.strip():
        st.error("题目、评分标准、答案三项都要填写哦。")
    else:
        t0 = time.time()
        with st.spinner("AI 正在阅卷…"):
            result, mode = grade_answer(question, criteria, answer, max_score)
        used = round(time.time() - t0, 2)
        st.session_state["last"] = {
            "student": student_name.strip() or "同学A",
            "answer": answer,
            "result": result,
            "mode": mode,
            "seconds": used,
            "time": dt.datetime.now().strftime("%H:%M:%S"),
        }

# ---------------- ③ AI 评分结果 + ④ 老师改分 ----------------
last = st.session_state.get("last")
if last:
    r = last["result"]
    st.markdown("### ③ AI 评分结果")
    m1, m2, m3 = st.columns(3)
    m1.metric("AI 给的分数", f"{r['score']} / {r['max_score']}")
    m2.metric("本次耗时", f"{last['seconds']} 秒")
    m3.metric("打分来源", "真实大模型" if last["mode"] == "deepseek" else "模拟")
    st.caption("总分由程序按逐条得分加总得出，不采用模型自报的总分（实测模型常把总计算错）")
    if r.get("model"):
        st.caption(f"使用模型：{r['model']}")

    st.write("**AI 的理由：**", r.get("reason", ""))

    if r.get("details"):
        st.write("**逐条明细：**")
        st.table([
            {
                "评分点": d.get("item", ""),
                "判定": d.get("level", ""),
                "得分": d.get("point", ""),
                "说明": d.get("comment", ""),
            }
            for d in r["details"]
        ])

    st.markdown("### ④ 老师改分并确认")
    with st.expander("展开学生答案原文（核对用）", expanded=False):
        st.code(last["answer"])
    c_a, c_b = st.columns([1, 1])
    with c_a:
        final_score = st.number_input("老师确认的分数", min_value=0,
                                      max_value=int(max_score), value=int(r["score"]))
    with c_b:
        st.write("")
        st.write("")
        if st.button("确认并记录", use_container_width=True):
            st.session_state.setdefault("records", []).append({
                "时间": last["time"],
                "学生": last["student"],
                "AI分数": r["score"],
                "老师分数": final_score,
                "满分": r["max_score"],
                "分差": round(float(r["score"]) - float(final_score), 1),
                "AI耗时(秒)": last["seconds"],
                "打分来源": "真实" if last["mode"] == "deepseek" else "模拟",
                "AI理由": r.get("reason", ""),
            })
            st.success(f"已记录：AI 打 {r['score']} 分，老师确认 {final_score} 分。")

# ---------------- 记录表 + 导出（做「效果验证」这张对比表用） ----------------
records = st.session_state.get("records", [])
if records:
    st.markdown("### ⑤ 评分记录（AI 与人工对比）")
    st.dataframe(records, use_container_width=True)

    import csv
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)
    st.download_button("下载记录 CSV（放进测试数据表）",
                       data=buf.getvalue().encode("utf-8-sig"),
                       file_name="ai_vs_human_records.csv",
                       mime="text/csv")
    st.caption("下载后用 Excel 打开，取各组平均分差和平均耗时，就是评审要的「AI 和人工对比数据」。")
