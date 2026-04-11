import os, time, re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
os.environ["GROQ_API_KEY"] = "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"  # ← key အစစ် ထည့်ပါ

# ════════════════════════════════════════
# TXT PARSER
# ════════════════════════════════════════
def parse_school_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    admission = {}
    for block in re.finditer(
        r'\[ADMISSION:(\d{4}-\d{4})\](.*?)(?=\[|\Z)', raw, re.DOTALL
    ):
        year, content = block.group(1), block.group(2)
        majors = {}
        for line in content.strip().splitlines():
            m = re.match(r'^\s*([\w\-]+)\s*=\s*(\d+)', line)
            if m:
                majors[m.group(1)] = int(m.group(2))
        if majors:
            admission[year] = majors

    opendate = {}
    for block in re.finditer(
        r'\[OPENDATE:(\d{4}-\d{4})\](.*?)(?=\[|\Z)', raw, re.DOTALL
    ):
        year, content = block.group(1), block.group(2)
        dates = {}
        for line in content.strip().splitlines():
            m = re.match(r'^\s*(sem\d)\s*=\s*(.+)', line)
            if m:
                dates[m.group(1)] = m.group(2).strip()
        if dates:
            opendate[year] = dates

    clean = re.sub(
        r'\[ADMISSION:.*?\].*?(?=\[|\Z)', '', raw, flags=re.DOTALL
    )
    clean = re.sub(
        r'\[OPENDATE:.*?\].*?(?=\[|\Z)', '', clean, flags=re.DOTALL
    )
    clean = re.sub(r'\[([\w:]+)\]', r'\n--- \1 ---\n', clean)
    clean = "\n".join(l for l in clean.splitlines() if l.strip())

    admission_text = "\n--- ADMISSION_SCORES ---\n"
    for year, majors in admission.items():
        admission_text += f"\n{year} ပညာသင်နှစ် ဝင်ခွင့်အမှတ်များ:\n"
        for major, score in majors.items():
            admission_text += f"  {major}: {score} မှတ်\n"

    opendate_text = "\n--- OPENING_DATES ---\n"
    for year, dates in opendate.items():
        opendate_text += f"\n{year} ကျောင်းဖွင့်ရက်များ:\n"
        opendate_text += f"  ပထမ Semester: {dates.get('sem1', '-')}\n"
        opendate_text += f"  ဒုတိယ Semester: {dates.get('sem2', '-')}\n"

    clean = clean + admission_text + opendate_text
    return admission, opendate, clean, list(admission.keys())


# ════════════════════════════════════════
# PYTHON MATH
# ════════════════════════════════════════
def calc_admission(score, year, admission):
    data   = admission[year]
    can    = sorted([(m,v) for m,v in data.items() if score >= v], key=lambda x: x[1])
    cannot = sorted([(m,v) for m,v in data.items() if score < v],  key=lambda x: x[1])

    if can:
        can_text    = "\n".join(f"• {m} (လိုအပ်သောအမှတ်: {v})" for m,v in can)
        cannot_text = "\n".join(f"• {m} ({v} မှတ် လိုသည်)" for m,v in cannot)
        result = (
            f"📋 {year} ပညာသင်နှစ် — အမှတ် {score}\n\n"
            f"✅ တက်နိုင်တဲ့ Major တွေ\n{can_text}"
        )
        if cannot:
            result += f"\n\n❌ အမှတ်မရောက်သေးတဲ့ Major တွေ\n{cannot_text}"
    else:
        min_v  = min(data.values())
        result = (
            f"📋 {year} ပညာသင်နှစ် — အမှတ် {score}\n\n"
            f"😔 အမှတ် {score} နဲ့တော့ ဘယ် Major မှ မတက်နိုင်သေးပါဘူး။\n"
            f"• အနည်းဆုံး {min_v} မှတ် ရမှ တက်နိုင်ပါမယ်။"
        )
    return result


def get_opendate(year, opendate):
    dates = opendate.get(year, {})
    sem1  = dates.get("sem1", "သတင်းအချက်အလက် မရှိသေးပါ")
    sem2  = dates.get("sem2", "သတင်းအချက်အလက် မရှိသေးပါ")
    return (
        f"📅 {year} ပညာသင်နှစ် ကျောင်းဖွင့်ရက်များ\n\n"
        f"• ပထမ Semester   — {sem1}\n"
        f"• ဒုတိယ Semester — {sem2}"
    )


def extract_score(text):
    scores = re.findall(r'\b([23456789]\d{2})\b', text)
    return int(scores[0]) if scores else None

def extract_year(text, years):
    m = re.search(r'(20\d{2}-20\d{2})', text)
    if m and m.group(1) in years:
        return m.group(1)
    for y in years:
        if y.split("-")[0] in text:
            return y
    return None


# ════════════════════════════════════════
# SCHOOL BOT
# ════════════════════════════════════════
class SchoolBot:
    def __init__(self):
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "school.txt"
        )
        self.admission, self.opendate, self.school_text, self.years = \
            parse_school_file(data_path)

        self.models = [
            {
                "name": "llama-3.1-8b-instant",
                "llm":  ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    max_tokens=1200
                ),
                "available": True, "reset_time": None
            },
            {
                "name": "llama-3.3-70b-versatile",
                "llm":  ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=1200
                ),
                "available": True, "reset_time": None
            },
        ]
        self.idx              = 0
        self.history          = []
        self.pending_score    = None
        self.pending_opendate = False

    def _get_llm(self):
        for m in self.models:
            if not m["available"] and m["reset_time"]:
                if time.time() > m["reset_time"]:
                    m["available"] = True
                    m["reset_time"] = None
        if self.models[self.idx]["available"]:
            return self.models[self.idx]
        for i, m in enumerate(self.models):
            if m["available"]:
                self.idx = i
                return m
        return None

    def _llm_call(self, msgs):
        attempts = 0
        while attempts < len(self.models):
            model = self._get_llm()
            if not model:
                raise Exception("AI တွေ busy ဖြစ်နေတယ်။ ခဏနေ ပြန်မေးပါနော်။")
            try:
                r = model["llm"].invoke(msgs)
                return r.content, model["name"]
            except Exception as e:
                err = str(e)
                if any(k in err for k in ["429","rate_limit","decommissioned"]):
                    mins  = 15
                    match = re.search(r'(\d+)m', err)
                    if match:
                        mins = int(match.group(1)) + 2
                    model["available"]  = False
                    model["reset_time"] = time.time() + mins * 60
                    self.idx = (self.idx + 1) % len(self.models)
                    attempts += 1
                else:
                    raise e
        raise Exception("All models failed")

    def _save(self, q, a):
        if self.history and self.history[-1].content == a:
            return
        self.history.append(HumanMessage(content=q))
        self.history.append(AIMessage(content=a))
        if len(self.history) > 12:
            self.history = self.history[-12:]

    def _year_prompt(self):
        opts = "\n".join(f"• {y}" for y in self.years)
        return f"ဘယ် ပညာသင်နှစ်အတွက် သိချင်တာလဲနော်?\n\n{opts}"

    def ask(self, question: str) -> dict:
        if not question.strip():
            return {"answer": "", "model": ""}

        score = extract_score(question)
        year  = extract_year(question, self.years)

        # Opening date path
        opendate_keywords = [
            "ကျောင်းဖွင့်", "ဖွင့်ရက်", "ဖွင့်မည့်", "ဖွင့်တဲ့",
            "semester စတင်", "ကျောင်းစ", "စတင်ဖွင့်"
        ]
        if any(k in question for k in opendate_keywords):
            self.pending_opendate = True
            self.pending_score    = None

        if self.pending_opendate:
            if year:
                ans = get_opendate(year, self.opendate)
                self.pending_opendate = False
                self._save(question, ans)
                return {"answer": ans, "model": "⚙️ Python"}
            ans = self._year_prompt()
            self._save(question, ans)
            return {"answer": ans, "model": "⚙️ Python"}

        # Threshold path
        threshold_keywords = [
            "ဝင်ခွင့်", "လိုတဲ့", "ရမှတ်", "လိုအပ်တဲ့",
            "minimum", "threshold", "ဘယ်လောက်ရမယ်",
            "ဘယ်နှစ်မှတ်", "အမှတ်ဘယ်လောက်"
        ]
        if any(k in question for k in threshold_keywords) and not score:
            self.pending_score = None
            if not year:
                ans = self._year_prompt()
                self._save(question, ans)
                return {"answer": ans, "model": "⚙️ Python"}
            data  = self.admission[year]
            lines = "\n".join(
                f"• {m}: {v} မှတ်"
                for m, v in sorted(data.items(), key=lambda x: x[1])
            )
            ans = f"📋 {year} ပညာသင်နှစ် ဝင်ခွင့်အမှတ်များ\n\n{lines}"
            self._save(question, ans)
            return {"answer": ans, "model": "⚙️ Python"}

        # Admission calc path
        if score:
            self.pending_score = score
        if self.pending_score:
            if year:
                ans = calc_admission(self.pending_score, year, self.admission)
                self.pending_score = None
                self._save(question, ans)
                return {"answer": ans, "model": "⚙️ Python"}
            ans = self._year_prompt()
            self._save(question, ans)
            return {"answer": ans, "model": "⚙️ Python"}

        # LLM path
        self.pending_score    = None
        self.pending_opendate = False

        system = f"""သင်သည် Polytechnic Maubin ကျောင်းရဲ့ AI Assistant ဖြစ်သည်။
သင်သည် ကျောင်းသားများနှင့် ဆရာတစ်ဦးကဲ့သို့ ရင်းနှီးစွာ ပြောဆိုရမည်။

ကျောင်းအချက်အလက်များ:
{self.school_text}

Response Style:
- ရင်းနှီးတဲ့ မြန်မာဘာသာနဲ့ ပြောပါ
- "ဟုတ်ကဲ့" "နော်" "ပါ" natural tone သုံးပါ
- list များကို • သုံးပြီး ခွဲပြပါ
- data မပါသောအချက် → "ကျောင်းကို တိုက်ရိုက်ဆက်သွယ်ပါနော် 📞 09665544356"
- ✅ subjects အမည်များ English အတိုင်းသာ သုံးပါ

ကိုင်တွယ်ရမည့်ပုံစံ:
ကျောင်းဝင်ကြေး မေးလျှင်    → ငွေပမာဏသာ ဖြေ
ကျောင်းအပ်ရက် မေးလျှင်    → ရက်စွဲသာ ဖြေ
စာရွက်စာတမ်း မေးလျှင်     → ပထမနှစ်လား အခြားနှစ်လား မေးမှ ဖြေ
အဆောင်ကြေး မေးလျှင်       → GENERAL section မှ ဖြေ
ဌာန/Major မေးလျှင်         → သေချာရှင်းပြ
Subject စာရင်း မေးလျှင်    → English အတိုင်းသာ ဖော်ပြပါ"""

        msgs = [SystemMessage(content=system)]
        msgs += self.history[-6:]
        msgs.append(HumanMessage(content=question))

        ans, model_name = self._llm_call(msgs)
        self._save(question, ans)
        return {"answer": ans, "model": model_name}

    def reset(self):
        self.history          = []
        self.pending_score    = None
        self.pending_opendate = False


# ════════════════════════════════════════
# SESSIONS (multi-user)
# ════════════════════════════════════════
sessions: dict = {}

def get_bot(session_id: str) -> SchoolBot:
    if session_id not in sessions:
        sessions[session_id] = SchoolBot()
    return sessions[session_id]