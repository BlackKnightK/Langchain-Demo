import os
import re
import json
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY1", "")

chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=5000,
    temperature=0
)

# Tavily key（联网工具用）
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY1", "").strip()
if not TAVILY_API_KEY:
    print("⚠️ 未检测到 TAVILY_API_KEY 环境变量。联网工具会失败，但离线工具仍可用。")


class SimpleLLMText:
    """把 ChatOpenAI 包一层，提供 generate(user_text, system_prompt=...) 以兼容 ReAct runner。"""
    def __init__(self, chat_model: ChatOpenAI):
        self.chat_model = chat_model

    def generate(self, user_text: str, system_prompt: str) -> str:
        msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_text)]
        resp = self.chat_model.invoke(msgs)
        return resp.content


llm_text = SimpleLLMText(chat_model)


# =============================
# 1) Tavily 底层搜索（统一入口）
# =============================
def tavily_search(query: str, search_depth: str = "basic", max_results: int = 5) -> dict:
    """统一的联网搜索底座：返回 Tavily 的原始 JSON。"""
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY 未配置")

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
    }
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


# =============================
# 2) 3 个联网工具（底层都是 tavily_search，输出规范化）
# =============================
def search_holiday_schedule(year: int, holiday: str) -> dict:
    """
    适用问题：
      - “YYYY 年某个节日怎么放假/调休？”
      - “清明/端午/中秋/国庆/春节放假安排？”
    输出字段：
      - year, holiday, summary, hints, sources
    """
    query = f"{year} {holiday} 放假 安排 调休"
    data = tavily_search(query=query, search_depth="basic", max_results=5)

    summary = (data.get("answer") or "").strip()
    results = data.get("results") or []
    sources = [{"title": x.get("title", ""), "url": x.get("url", "")} for x in results[:3]]

    return {
        "year": year,
        "holiday": holiday,
        "summary": summary if summary else "未在搜索结果中得到明确总结，请查看 sources。",
        "hints": "建议优先以权威来源为准（如官方通知/权威媒体）。",
        "sources": sources
    }


def search_weather(city: str, date: str = "today") -> dict:
    """
    适用问题：
      - “今天/明天/某日 某城市天气？”
      - “穿衣建议需要当天温度/降水信息”
    输出字段：
      - city, date, summary, sources
    """
    query = f"{city} {date} 天气 最高气温 最低气温 降水"
    data = tavily_search(query=query, search_depth="basic", max_results=5)

    summary = (data.get("answer") or "").strip()
    results = data.get("results") or []
    sources = [{"title": x.get("title", ""), "url": x.get("url", "")} for x in results[:3]]

    return {
        "city": city,
        "date": date,
        "summary": summary if summary else "未在搜索结果中得到明确总结，请查看 sources。",
        "sources": sources
    }


def search_exam_deadline(exam: str, session: str = "最近一次") -> dict:
    """
    适用问题：
      - “雅思/托福/四六级 报名截止时间？”
      - “某考试某月场次报名/截止/入口”
    输出字段：
      - exam, session, summary, sources, notes
    """
    query = f"{exam} {session} 报名 截止 时间"
    data = tavily_search(query=query, search_depth="basic", max_results=5)

    summary = (data.get("answer") or "").strip()
    results = data.get("results") or []
    sources = [{"title": x.get("title", ""), "url": x.get("url", "")} for x in results[:3]]

    return {
        "exam": exam,
        "session": session,
        "summary": summary if summary else "未在搜索结果中得到明确总结，请查看 sources。",
        "notes": "最终请以报名平台/官方页面为准（可能存在地区或场次差异）。",
        "sources": sources
    }


# =============================
# 3) 3 个离线工具（稳定、若使用本地API则无需联网）
# =============================
def calc(expression: str) -> dict:
    """
    适用问题：
      - 费用/时间/比例计算
      - “18 元每天一杯，一个月花多少？”
    输出字段：
      - expression, result
    """
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expression):
        return {"expression": expression, "result": None, "error": "表达式包含非法字符"}
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "result": None, "error": str(e)}


def time_plan(total_minutes: int, tasks: str) -> dict:
    """
    适用问题：
      - 时间安排/学习计划（短时段可执行）
    输入：
      - total_minutes: 总时长（分钟）
      - tasks: 任务描述（文本）
    输出字段：
      - total_minutes, plan (list), notes
    """
    parts = re.split(r"[，,；;、\n]+", tasks.strip())
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return {"total_minutes": total_minutes, "plan": [], "notes": ["未解析到任务"]}

    slot = max(5, total_minutes // len(parts))
    plan = []
    start = 0
    for p in parts:
        end = min(total_minutes, start + slot)
        plan.append({"task": p, "start_min": start, "end_min": end})
        start = end
        if start >= total_minutes:
            break

    return {
        "total_minutes": total_minutes,
        "plan": plan,
        "notes": ["这是粗分配骨架；若任务自带时长，建议按任务时长优先。"]
    }


def rewrite_text(text: str, style: str = "更礼貌") -> dict:
    """
    适用问题：
      - 改写/润色（更正式/更礼貌/更简洁）
    输出字段：
      - style, rewritten
    """
    if style in ("更礼貌", "礼貌", "polite"):
        rewritten = f"老师您好，{text}。给您带来不便非常抱歉，谢谢理解。"
    elif style in ("更正式", "正式", "formal"):
        rewritten = f"{text}。如需我提供进一步说明或材料，请您告知。"
    elif style in ("更简洁", "简洁", "concise"):
        rewritten = text.strip().rstrip("。.!") + "。"
    else:
        rewritten = text

    return {"style": style, "rewritten": rewritten}


# =============================
# 4) 工具注册表 + 工具说明（运行时打印）
# =============================
TOOLS = {
    "search_holiday_schedule": search_holiday_schedule,
    "search_weather": search_weather,
    "search_exam_deadline": search_exam_deadline,
    "calc": calc,
    "time_plan": time_plan,
    "rewrite_text": rewrite_text,
}

TOOL_HELP = """
【工具说明：本 Agent 可用 6 个工具】

联网工具（底层都调用 tavily_search，但做了不同“意图封装”和“输出字段规范”）：
1) search_holiday_schedule(year: int, holiday: str) -> dict
   - 适用问题：节假日放假/调休安排（例如“2026 清明怎么放假”）
   - 输出：year, holiday, summary, sources

2) search_weather(city: str, date: str="today") -> dict
   - 适用问题：今天/明天/某日天气 + 穿衣建议需要的事实
   - 输出：city, date, summary, sources

3) search_exam_deadline(exam: str, session: str="最近一次") -> dict
   - 适用问题：雅思/托福/四六级等报名截止时间
   - 输出：exam, session, summary, sources, notes

离线工具（无需联网，稳定可用）：
4) calc(expression: str) -> dict
   - 适用问题：费用/时间/比例计算
   - 输出：expression, result

5) time_plan(total_minutes: int, tasks: str) -> dict
   - 适用问题：时间安排/学习计划（短时段可执行）
   - 输出：plan（分钟区间）

6) rewrite_text(text: str, style: str="更礼貌") -> dict
   - 适用问题：改写/润色（礼貌/正式/简洁）
   - 输出：rewritten
"""


# =============================
# 5) Agent System Prompt（强约束输出）
# =============================
AGENT_SYSTEM_PROMPT = f"""你是一个“校园生活助手”Agent。你可以调用工具来完成任务。
你的目标：根据用户问题选择合适工具，必要时多步调用，最后输出对用户友好的最终答案。

可用工具（名称必须完全匹配）：
- search_holiday_schedule(year, holiday)
- search_weather(city, date)
- search_exam_deadline(exam, session)
- calc(expression)
- time_plan(total_minutes, tasks)
- rewrite_text(text, style)

重要规则（必须遵守）：
1) 你每次只能输出以下两种形式之一：
   - Action: 工具名(参数名="参数值", ...)
   - Action: finish(answer="最终给用户的自然语言回答")
2) 当问题需要“最新公开信息”（节假日安排/天气/报名截止），优先使用对应的联网工具。
3) 当问题属于计算/规划/改写，使用离线工具，不要联网。
4) 所有参数尽量用双引号包裹（即使是数字也可以用引号），避免解析失败。
5) 若需要多步：先 Action 调工具，等 Observation 后再决定下一步。
6) 最终回答要自然语言、可直接给人看，并在末尾附上一行“本次工具链：...”。

{TOOL_HELP}
"""


# =============================
# 6) ReAct 执行器（支持一轮多 Action）
# =============================
def _parse_actions(llm_output: str):
    """提取所有 Action 行（支持一轮多条 Action）。"""
    actions = []
    for line in llm_output.splitlines():
        line = line.strip()
        if line.startswith("Action:") or line.startswith("Action："):
            actions.append(line.split(":", 1)[-1].strip() if "Action:" in line else line.split("：", 1)[-1].strip())
    return actions


def _parse_tool_call(action_str: str):
    """解析 tool_name(...)，支持 key="v" / key='v' / key=123 / key=true 等。"""
    m = re.match(r"(\w+)\s*\(\s*(.*)\s*\)\s*$", action_str)
    if not m:
        return None, None

    tool_name = m.group(1).strip()
    args_str = m.group(2).strip()

    kwargs = {}

    # 1) 带双引号
    for k, v in re.findall(r'(\w+)\s*=\s*"([^"]*)"', args_str):
        kwargs[k] = v

    # 2) 带单引号
    for k, v in re.findall(r"(\w+)\s*=\s*'([^']*)'", args_str):
        kwargs[k] = v

    # 3) 数值不带引号：key=123 / key=-3 / key=3.14
    for k, v in re.findall(r'(\w+)\s*=\s*([-+]?\d+(?:\.\d+)?)', args_str):
        if k in kwargs:
            continue
        kwargs[k] = float(v) if "." in v else int(v)

    # 4) bool 不带引号：key=true/false
    for k, v in re.findall(r'(\w+)\s*=\s*(true|false)', args_str, flags=re.IGNORECASE):
        if k in kwargs:
            continue
        kwargs[k] = (v.lower() == "true")

    return tool_name, kwargs


def run_react_agent(user_prompt: str, max_loops: int = 6):
    prompt_history = [f"用户请求: {user_prompt}"]
    tool_chain = []

    print(f"\n用户输入: {user_prompt}\n" + "=" * 70)

    for i in range(max_loops):
        print(f"--- 循环 {i+1} ---")

        full_prompt = "\n".join(prompt_history)
        llm_output = llm_text.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        print(f"模型输出:\n{llm_output}\n")

        prompt_history.append(llm_output)

        actions = _parse_actions(llm_output)

        # ✅ 兜底：模型直接输出最终答案但没有 Action 行
        if not actions:
            final_answer = llm_output.strip()
            # 若模型没写工具链，就补一行
            if "本次工具链" not in final_answer:
                final_answer += "\n\n本次工具链：" + (" → ".join(tool_chain) if tool_chain else "(无)")
            print("\n任务完成（兜底），最终答案:\n" + final_answer)
            return final_answer

        for idx, action_str in enumerate(actions, 1):
            print(f"解析到的 Action[{idx}]: {action_str}")

            # finish
            if action_str.lower().startswith("finish"):
                m = re.search(r'answer\s*=\s*"([\s\S]*?)"\s*\)\s*$', action_str, re.IGNORECASE)
                final_answer = m.group(1) if m else action_str
                if "本次工具链" not in final_answer:
                    final_answer = final_answer.rstrip() + "\n\n本次工具链：" + (" → ".join(tool_chain) if tool_chain else "(无)")
                print("\n任务完成，最终答案:\n" + final_answer)
                return final_answer

            tool_name, kwargs = _parse_tool_call(action_str)
            if not tool_name:
                obs = f"工具调用解析失败: {action_str}"
                print(f"Observation: {obs}")
                prompt_history.append(f"Observation: {obs}")
                continue

            if tool_name not in TOOLS:
                obs = f"未定义的工具: {tool_name}"
                print(f"Observation: {obs}")
                prompt_history.append(f"Observation: {obs}")
                continue

            # 最小类型转换
            try:
                for k in list(kwargs.keys()):
                    if k in ("year", "total_minutes"):
                        kwargs[k] = int(kwargs[k])
            except Exception:
                pass

            try:
                result = TOOLS[tool_name](**kwargs)
                tool_chain.append(tool_name)
                obs = json.dumps(result, ensure_ascii=False)
            except Exception as e:
                obs = f"工具执行失败: {e}"

            print(f"Observation: {obs}")
            print("-" * 70)
            prompt_history.append(f"Observation: {obs}")

    print("\n达到最大循环次数，任务可能未完成。")
    return None


# =============================
# 7) 交互式入口
# =============================
if __name__ == "__main__":
    print(TOOL_HELP)
    print("\n可直接复制这些测试输入：")
    print("1) 2026 年清明节放假怎么安排？给出日期范围并一句话概括。")
    print("2) 今天上海天气怎么样？一句话概括并给穿衣建议。")
    print("3) 2026 年 6 月雅思报名截止时间大概是什么时候？给出日期并提醒要准备什么材料。")
    print("4) 一杯奶茶 18 元，我每天一杯，一个月（30 天）花多少钱？")
    print("5) 我今晚只有 90 分钟，要阅读40、背单词30、剩下时间休息，帮我排一个顺序。")
    print("6) 把这句话改得更礼貌一点：老师我明天不来上课。")

    while True:
        q = input("\n请输入问题（输入 退出 结束）：").strip()
        if q in ("退出", "quit", "exit"):
            break
        run_react_agent(q, max_loops=6)
