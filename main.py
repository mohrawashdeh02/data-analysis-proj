from fastapi import FastAPI, File, UploadFile, Request, Form, Cookie
from fastapi.responses import HTMLResponse, JSONResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional
import pandas as pd
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os
import uuid
import httpx
import json
try:
    import openpyxl
except ImportError:
    pass

# ─── Configuration ───────────────────────────────────────────────────────────
# احصل على مفتاح مجاني: https://console.groq.com
# export GROQ_API_KEY="gsk_..."
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.1-8b-instant"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

for folder in ("static", "uploads"):
    os.makedirs(folder, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

_sessions: dict[str, pd.DataFrame] = {}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_df(session_id: Optional[str]):
    if not session_id:
        return None
    return _sessions.get(session_id)


def _fig_to_base64() -> str:
    buf = io.BytesIO()
    try:
        plt.tight_layout(pad=2.0)
    except Exception:
        pass
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return encoded


def _auto_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    for col in df_out.columns:
        if not pd.api.types.is_numeric_dtype(df_out[col]):
            cleaned = df_out[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            converted = pd.to_numeric(cleaned, errors='coerce')
            if converted.notna().sum() > len(df_out) * 0.5:
                df_out[col] = converted
    return df_out


# ─── AI Analysis via Anthropic API ───────────────────────────────────────────

async def ai_analyze_arabic(prompt: str) -> str:
    """Call DeepSeek API for intelligent Arabic analysis."""
    if not GROQ_API_KEY:
        return "⚠️ لم يتم تعيين GROQ_API_KEY. الرجاء تعيين متغير البيئة."
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json={
                    "model": GROQ_MODEL,
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            data = response.json()
            if response.status_code != 200:
                return f"خطأ من API: {data.get('error', {}).get('message', str(data))}"
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
            return f"لا توجد استجابة: {data}"
    except Exception as e:
        return f"خطأ في التحليل: {str(e)}"


async def analyze_column_arabic(df: pd.DataFrame, column: str) -> str:
    col = df[column]
    if pd.api.types.is_numeric_dtype(col):
        stats = col.describe()
        prompt = f"""أنت محلل بيانات خبير. بناءً على الإحصائيات التالية للعمود "{column}":
- عدد القيم: {int(stats['count'])}
- المتوسط: {stats['mean']:.2f}
- الحد الأدنى: {stats['min']:.2f}
- الحد الأقصى: {stats['max']:.2f}
- الانحراف المعياري: {stats['std']:.2f}

اكتب فقرة قصيرة باللغة العربية (3-4 جمل) تشرح فيها توزيع هذا العمود وأهم ما يلفت الانتباه. كن ذكياً ومحدداً."""
    else:
        top = col.value_counts().head(5)
        top_str = "\n".join([f"- {k}: {v} مرة" for k, v in top.items()])
        prompt = f"""أنت محلل بيانات خبير. بناءً على أكثر القيم تكراراً في العمود "{column}":
{top_str}
(إجمالي القيم الفريدة: {col.nunique()})

اكتب فقرة قصيرة باللغة العربية (3-4 جمل) تشرح فيها أبرز ما تلاحظه في هذا العمود."""
    return await ai_analyze_arabic(prompt)


async def analyze_relationship_arabic(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    x_numeric = pd.api.types.is_numeric_dtype(df[x_col])
    y_numeric = pd.api.types.is_numeric_dtype(df[y_col])

    if x_numeric and y_numeric:
        clean_df = df[[x_col, y_col]].dropna()
        corr = clean_df[x_col].corr(clean_df[y_col])
        prompt = f"أنت محلل بيانات خبير. حلل العلاقة بين {x_col} و {y_col} (معامل الارتباط: {corr:.3f}) بفقرة عربية موجزة."
    else:
        prompt = f"أنت محلل بيانات خبير. اشرح العلاقة الظاهرة بين الفئات في {x_col} والقيم في {y_col} بفقرة عربية موجزة."

    return await ai_analyze_arabic(prompt)


async def analyze_heatmap_arabic(df: pd.DataFrame) -> str:
    numeric_df = df.select_dtypes(include="number")
    prompt = f"""أنت محلل بيانات خبير. حلل مصفوفة الارتباط للأعمدة الرقمية التالية: {list(numeric_df.columns)}.
اشرح بفقرة عربية قصيرة أي المتغيرات تؤثر على بعضها بشكل أقوى."""
    return await ai_analyze_arabic(prompt)


# ─── Rendering ───────────────────────────────────────────────────────────────


# ─── Data Quality Check ───────────────────────────────────────────────────────

def data_quality_check(df: pd.DataFrame) -> list:
    report = []
    for col in df.columns:
        info = {"col": col, "issues": []}
        missing_pct = df[col].isna().mean() * 100
        info["missing_pct"] = round(missing_pct, 1)
        if missing_pct > 30:
            info["status"] = "🔴"; info["issues"].append(f"قيم ناقصة {missing_pct:.0f}%")
        elif missing_pct > 5:
            info["status"] = "🟡"; info["issues"].append(f"قيم ناقصة {missing_pct:.0f}%")
        else:
            info["status"] = "🟢"
        info["outliers"] = 0
        if pd.api.types.is_numeric_dtype(df[col]):
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = int(((df[col] < q1-1.5*iqr) | (df[col] > q3+1.5*iqr)).sum())
                info["outliers"] = outliers
                if outliers > 0:
                    info["issues"].append(f"{outliers} قيمة شاذة")
                    if info["status"] == "🟢": info["status"] = "🟡"
        info["issues_str"] = " | ".join(info["issues"]) if info["issues"] else "سليم ✓"
        report.append(info)
    return report


# ─── Calculate KPIs (Python, 100% accurate) ───────────────────────────────────

def calculate_kpis(df: pd.DataFrame) -> list:
    kpis = []
    cols_lower = {c.lower(): c for c in df.columns}
    def get_col(*names):
        for n in names:
            if n.lower() in cols_lower: return cols_lower[n.lower()]
        return None

    att_col = get_col("attrition","turnover","left","resigned")
    if att_col:
        rate = df[att_col].astype(str).str.lower().isin(["yes","1","true","left"]).mean()*100
        kpis.append({"name":"Attrition Rate","value":f"{rate:.1f}%","raw":rate})

    sal_col = get_col("monthlyincome","salary","income","wage","pay","monthly_income")
    if sal_col and pd.api.types.is_numeric_dtype(df[sal_col]):
        avg = df[sal_col].mean()
        kpis.append({"name":"Avg Monthly Income","value":f"{avg:,.0f}","raw":avg,
                     "extra":f"min={df[sal_col].min():,.0f} max={df[sal_col].max():,.0f}"})

    ot_col = get_col("overtime","over_time","ot")
    if ot_col:
        if pd.api.types.is_numeric_dtype(df[ot_col]):
            kpis.append({"name":"Avg Overtime Hours","value":f"{df[ot_col].mean():.1f} hrs","raw":df[ot_col].mean()})
        else:
            ot_rate = df[ot_col].astype(str).str.lower().isin(["yes","1","true"]).mean()*100
            kpis.append({"name":"Overtime Rate","value":f"{ot_rate:.1f}%","raw":ot_rate})

    ten_col = get_col("yearsatcompany","tenure","years_at_company","totalworkingyears")
    if ten_col and pd.api.types.is_numeric_dtype(df[ten_col]):
        kpis.append({"name":"Avg Tenure (Years)","value":f"{df[ten_col].mean():.1f} yrs","raw":df[ten_col].mean()})

    age_col = get_col("age","employee_age")
    if age_col and pd.api.types.is_numeric_dtype(df[age_col]):
        kpis.append({"name":"Avg Employee Age","value":f"{df[age_col].mean():.1f} yrs","raw":df[age_col].mean()})

    sat_col = get_col("jobsatisfaction","satisfaction","satisfactionlevel")
    if sat_col and pd.api.types.is_numeric_dtype(df[sat_col]):
        kpis.append({"name":"Avg Job Satisfaction","value":f"{df[sat_col].mean():.1f}/{int(df[sat_col].max())}","raw":df[sat_col].mean()})

    perf_col = get_col("performancerating","performance","rating")
    if perf_col and pd.api.types.is_numeric_dtype(df[perf_col]):
        kpis.append({"name":"Avg Performance Rating","value":f"{df[perf_col].mean():.1f}/{int(df[perf_col].max())}","raw":df[perf_col].mean()})

    wlb_col = get_col("worklifebalance","work_life_balance")
    if wlb_col and pd.api.types.is_numeric_dtype(df[wlb_col]):
        kpis.append({"name":"Work-Life Balance","value":f"{df[wlb_col].mean():.1f}/{int(df[wlb_col].max())}","raw":df[wlb_col].mean()})

    dept_col = get_col("department","dept","division","team")
    if dept_col:
        top = df[dept_col].value_counts().idxmax()
        pct = df[dept_col].value_counts(normalize=True).max()*100
        kpis.append({"name":"Largest Department","value":str(top),"raw":pct,"extra":f"{pct:.0f}% of workforce"})

    gen_col = get_col("gender","sex")
    if gen_col:
        counts = df[gen_col].value_counts(normalize=True)*100
        parts = [f"{k}: {v:.0f}%" for k,v in counts.head(2).items()]
        kpis.append({"name":"Gender Distribution","value":" | ".join(parts),"raw":0})

    return kpis


# ─── Build Period Comparison ──────────────────────────────────────────────────

def build_period_comparison(df: pd.DataFrame) -> dict:
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = None
    for kw in ["date","month","year","period","quarter","time"]:
        for c in df.columns:
            if kw in c.lower(): date_col = c; break
        if date_col: break

    if date_col:
        try:
            df2 = df.copy()
            df2["_date"] = pd.to_datetime(df2[date_col], errors="coerce")
            med = df2["_date"].median()
            pa = df2[df2["_date"] <= med]; pb = df2[df2["_date"] > med]
            la = f"الفترة الأولى (قبل {med.strftime('%Y-%m')})"
            lb = f"الفترة الثانية (بعد {med.strftime('%Y-%m')})"
        except: date_col = None

    if not date_col:
        split_col = None
        for name in ["yearsatcompany","employeenumber","totalworkingyears","tenure"]:
            if name in cols_lower: split_col = cols_lower[name]; break
        if split_col:
            med = df[split_col].median()
            pa = df[df[split_col] <= med]; pb = df[df[split_col] > med]
            la = f"المجموعة الأولى ({split_col} ≤ {med:.0f})"
            lb = f"المجموعة الثانية ({split_col} > {med:.0f})"
        else:
            mid = len(df)//2; pa = df.iloc[:mid]; pb = df.iloc[mid:]
            la = "النصف الأول"; lb = "النصف الثاني"

    ka = calculate_kpis(pa); kb = calculate_kpis(pb)
    da = {k["name"]: k for k in ka}; db = {k["name"]: k for k in kb}
    rows = []
    for name in da:
        if name in db:
            a=da[name]; b=db[name]; ra=a.get("raw",0); rb=b.get("raw",0)
            if isinstance(ra,(int,float)) and isinstance(rb,(int,float)) and ra!=0:
                chg=((rb-ra)/abs(ra))*100; direction="↑" if rb>ra else ("↓" if rb<ra else "→")
            else: chg=0; direction="→"
            rows.append({"name":name,"value_a":a["value"],"value_b":b["value"],
                         "raw_a":ra,"raw_b":rb,"change_pct":round(chg,1),"direction":direction})
    return {"label_a":la,"label_b":lb,"rows":rows,"n_a":len(pa),"n_b":len(pb)}


# ─── AI: Interpret KPIs ───────────────────────────────────────────────────────

async def interpret_kpis(kpis: list) -> list:
    if not kpis: return []
    kpi_text = "\n".join([f"- {k['name']}: {k['value']}"+(f" ({k.get('extra','')})" if k.get('extra') else "") for k in kpis])
    prompt = f"""Senior HR analytics expert. Interpret pre-calculated KPIs only.
Be specific. Write Arabic text fields in Arabic.
KPIs:\n{kpi_text}
Return ONLY valid JSON array (no markdown):
[{{"name":"exact name","value":"exact value","interpretation":"ماذا يعني (Arabic, 1 sentence)","risk":"High","recommendation":"إجراء مباشر (Arabic, 1 sentence)"}}]
Risk: High|Medium|Low. Return ALL {len(kpis)} KPIs."""
    raw = await ai_analyze_arabic(prompt)
    try:
        cleaned = raw.strip()
        if "```" in cleaned:
            for p in cleaned.split("```"):
                p=p.strip()
                if p.startswith("json"): p=p[4:].strip()
                if p.startswith("["): cleaned=p; break
        s=cleaned.find("["); e=cleaned.rfind("]")+1
        if s!=-1 and e>s: cleaned=cleaned[s:e]
        interpreted = json.loads(cleaned)
        result = []
        for i,kpi in enumerate(kpis):
            item = interpreted[i] if i<len(interpreted) else {}
            item["value"]=kpi["value"]; item.setdefault("name",kpi["name"])
            item.setdefault("risk","Medium"); item.setdefault("interpretation",""); item.setdefault("recommendation","")
            result.append(item)
        return result
    except: return [{"name":k["name"],"value":k["value"],"interpretation":"","risk":"Medium","recommendation":""} for k in kpis]


# ─── AI: Interpret Trends ────────────────────────────────────────────────────

async def interpret_kpi_trends(comparison: dict) -> dict:
    rows = comparison["rows"]
    if not rows: return {"kpi_trends":[],"summary":"","top_action":""}
    lines = []
    for r in rows:
        sign="+" if r["change_pct"]>0 else ""
        lines.append(f"- {r['name']}: {r['value_a']} → {r['value_b']} ({r['direction']} {sign}{r['change_pct']:.1f}%)")
    prompt = f"""Senior HR analytics expert. Analyze KPI trends.
Period A ({comparison['label_a']}): {comparison['n_a']} employees
Period B ({comparison['label_b']}): {comparison['n_b']} employees
Changes:\n{"\n".join(lines)}
Return ONLY valid JSON (no markdown):
{{"kpi_trends":[{{"name":"KPI","change":"التغيير (Arabic)","interpretation":"التأثير (Arabic)","risk":"High","recommendation":"الإجراء (Arabic)"}}],"summary":"ملخص (Arabic)","top_action":"أهم إجراء (Arabic)"}}
Only KPIs with >3% change. Sort High first."""
    raw = await ai_analyze_arabic(prompt)
    try:
        cleaned = raw.strip()
        if "```" in cleaned:
            for p in cleaned.split("```"):
                p=p.strip()
                if p.startswith("json"): p=p[4:].strip()
                if p.startswith("{"): cleaned=p; break
        s=cleaned.find("{"); e=cleaned.rfind("}")+1
        if s!=-1 and e>s: cleaned=cleaned[s:e]
        result=json.loads(cleaned)
        result.setdefault("kpi_trends",[]); result.setdefault("summary",""); result.setdefault("top_action","")
        for t in result["kpi_trends"]: t.setdefault("risk","Medium")
        return result
    except: return {"kpi_trends":[],"summary":"","top_action":""}


# ─── AI: Smart Analysis ───────────────────────────────────────────────────────

async def generate_smart_analysis(df: pd.DataFrame, quality: list) -> dict:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = list(df.columns)
    stats = " | ".join([f"{c}: mean={df[c].mean():.1f} min={df[c].min():.1f} max={df[c].max():.1f}" for c in num_cols[:6] if len(df[c].dropna())])
    corr_lines = []
    if len(num_cols)>=2:
        cm=df[num_cols].corr()
        pairs=[(num_cols[i],num_cols[j],round(cm.iloc[i,j],2)) for i in range(len(num_cols)) for j in range(i+1,len(num_cols)) if abs(cm.iloc[i,j])>0.35]
        pairs.sort(key=lambda x:abs(x[2]),reverse=True)
        corr_lines=[f"{a}&{b}={v}" for a,b,v in pairs[:4]]
    quality_issues=[f"{i['col']}: {i['issues_str']}" for i in quality if i["issues"]]
    prompt = f"""Senior HR analytics expert. Analyze dataset.
Dataset: Rows={len(df)}, Columns={all_cols}
Stats: {stats or "none"} | Correlations: {" | ".join(corr_lines) or "none"}
Quality: {" | ".join(quality_issues) or "none"}
Return ONLY JSON (no markdown):
{{"insights":["ملاحظة 1 (Arabic)","ملاحظة 2","ملاحظة 3"],"warnings":[],"recommendations":[{{"observation":"نمط (Arabic)","impact":"تأثير (Arabic)","action":"إجراء (Arabic)","priority":"High","plot_action":"plot_column","col":"EXACT_COL","col2":null}}]}}
Rules: 3 insights | 2-4 recs | col exact from {all_cols} | plot_action: plot_column/plot_relation/heatmap"""
    raw = await ai_analyze_arabic(prompt)
    try:
        cleaned=raw.strip()
        if "```" in cleaned:
            for p in cleaned.split("```"):
                p=p.strip()
                if p.startswith("json"): p=p[4:].strip()
                if p.startswith("{"): cleaned=p; break
        s=cleaned.find("{"); e=cleaned.rfind("}")+1
        if s!=-1 and e>s: cleaned=cleaned[s:e]
        result=json.loads(cleaned)
        result.setdefault("insights",[]); result.setdefault("warnings",[]); result.setdefault("recommendations",[])
        valid=[]
        for rec in result["recommendations"]:
            col=rec.get("col"); col2=rec.get("col2")
            if col and col not in all_cols: continue
            if col2 and col2 not in all_cols: rec["col2"]=None; rec["plot_action"]="plot_column"
            rec.setdefault("priority","Medium"); valid.append(rec)
        result["recommendations"]=valid
        return result
    except: return {"insights":[raw[:200] if raw else "تعذّر التحليل."],"warnings":[],"recommendations":[]}


# ─── AI: Dashboard-Style HR Report ───────────────────────────────────────────

async def generate_hr_report(kpis: list, analysis: dict, quality: list) -> str:
    kpi_text     = "\n".join([f"- {k['name']}: {k['value']}"+(f" ({k.get('extra','')})" if k.get('extra') else "") for k in kpis])
    insights_text= "\n".join([f"- {i}" for i in analysis.get("insights",[])])
    recs_text    = "\n".join([f"- {r.get('action','')}" for r in analysis.get("recommendations",[])])
    quality_issues= [f"- {i['col']}: {i['issues_str']}" for i in quality if i["issues"]]
    quality_text = "\n".join(quality_issues) if quality_issues else "- لا توجد مشاكل"

    prompt = f"""You are a senior HR analytics expert and product designer.
Present HR analysis in a clear dashboard format. Write everything in Arabic.
Be concise — no long paragraphs.

DATA:
KPIs: {kpi_text}
Insights: {insights_text}
Recommendations: {recs_text}
Data Issues: {quality_text}

OUTPUT FORMAT (exact headers in Arabic):

## بطاقات الأداء KPI CARDS
لكل KPI: الاسم | القيمة | الحالة (🔴 High / 🟡 Medium / 🟢 Low) | المعنى (جملة واحدة)

## 🔥 أهم إجراء TOP ACTION
توصية واحدة — الأعلى تأثيراً — محددة ومباشرة

## 📊 أبرز النتائج KEY FINDINGS
- 3 نقاط فقط، كل نقطة جملة واحدة قصيرة

## ⚠️ المخاطر RISKS
- قائمة قصيرة بأهم المخاطر

## 💡 التوصيات RECOMMENDATIONS
- 3 إلى 5 إجراءات محددة وعملية فقط"""

    return await ai_analyze_arabic(prompt)


def _base_context(df: Optional[pd.DataFrame], **extra):
    return {
        "columns": list(df.columns) if df is not None else None,
        "numeric_columns": df.select_dtypes(include="number").columns.tolist() if df is not None else [],
        "all_columns": list(df.columns) if df is not None else [],
        "summary": df.describe().to_html(classes="table table-striped") if df is not None else None,
        "head_table": df.head(10).to_html(
            classes="table table-striped table-hover", index=False
        ) if df is not None else None,
        "plot": None,
        "error": None,
        "ai_analysis": None,
        "quality_report": None,
        "smart_analysis": None,
        "kpis": None,
        "kpi_trends": None,
        **extra,
    }


def _render(request: Request, df: Optional[pd.DataFrame], **extra):
    context = _base_context(df, **extra)
    return templates.TemplateResponse(request=request, name="index.html", context=context)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, session_id: Optional[str] = Cookie(default=None)):
    df = _get_df(session_id)
    return _render(request, df)


@app.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    session_id: Optional[str] = Cookie(default=None),
):
    try:
        if not file.filename.lower().endswith((".csv", ".txt")):
            raise ValueError("يرجى رفع ملف CSV فقط")

        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("حجم الملف كبير جداً (الحد الأقصى 10 ميجابايت)")

        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            df = pd.read_csv(io.BytesIO(content), encoding="ISO-8859-1")

        if df.empty:
            raise ValueError("الملف فارغ")

        if not session_id:
            session_id = str(uuid.uuid4())

        _sessions[session_id] = df
        import asyncio
        quality    = data_quality_check(df)
        raw_kpis   = calculate_kpis(df)
        comparison = build_period_comparison(df)
        analysis, interpreted_kpis, trends = await asyncio.gather(
            generate_smart_analysis(df, quality),
            interpret_kpis(raw_kpis),
            interpret_kpi_trends(comparison)
        )
        response = _render(request, df,
                           quality_report=quality,
                           smart_analysis=analysis,
                           kpis=interpreted_kpis,
                           kpi_trends={"comparison": comparison, "trends": trends})
        response.set_cookie("session_id", session_id)
        return response

    except Exception as e:
        return _render(request, None, error=str(e))


@app.post("/plot_column", response_class=HTMLResponse)
async def plot_column(
    request: Request,
    column: str = Form(...),
    session_id: Optional[str] = Cookie(default=None),
):
    df = _get_df(session_id)
    if df is None:
        return _render(request, None, error="لم يتم تحميل أي ملف")
    try:
        plt.figure(figsize=(10, 5))
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column].plot(kind="hist", bins=20, color="skyblue", edgecolor="black")
        else:
            df[column].value_counts().head(20).plot(kind="bar", color="salmon")

        plot_b64 = _fig_to_base64()
        ai_text = await analyze_column_arabic(df, column)
        return _render(request, df, plot=plot_b64, ai_analysis=ai_text)
    except Exception as e:
        plt.close()
        return _render(request, df, error=str(e))


@app.post("/plot_relation", response_class=HTMLResponse)
async def plot_relation(
    request: Request,
    x_col: str = Form(...),
    y_col: str = Form(...),
    session_id: Optional[str] = Cookie(default=None),
):
    df = _get_df(session_id)
    if df is None:
        return _render(request, None, error="لم يتم تحميل أي ملف")
    try:
        df_work = _auto_numeric(df)
        x_numeric = pd.api.types.is_numeric_dtype(df_work[x_col])
        y_numeric = pd.api.types.is_numeric_dtype(df_work[y_col])

        plt.figure(figsize=(10, 6))

        if x_numeric and y_numeric:
            plt.scatter(
                df_work[x_col], df_work[y_col],
                alpha=0.6, color="steelblue", edgecolors="white", linewidth=0.5,
            )
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"العلاقة بين {x_col} و {y_col}")

        elif not x_numeric and y_numeric:
            group = (
                df_work.groupby(x_col)[y_col]
                .mean()
                .sort_values(ascending=False)
                .head(20)
            )
            group.plot(kind="bar", color="coral", edgecolor="black")
            plt.xlabel(x_col)
            plt.ylabel(f"متوسط {y_col}")
            plt.title(f"متوسط {y_col} حسب {x_col}")
            plt.xticks(rotation=45, ha="right")

        elif x_numeric and not y_numeric:
            categories = df_work[y_col].dropna().unique()[:10]
            data_to_plot = [
                df_work[df_work[y_col] == cat][x_col].dropna().values
                for cat in categories
            ]
            plt.boxplot(data_to_plot, labels=categories)
            plt.xlabel(y_col)
            plt.ylabel(x_col)
            plt.title(f"توزيع {x_col} حسب {y_col}")
            plt.xticks(rotation=45, ha="right")

        else:
            cross = pd.crosstab(df_work[x_col], df_work[y_col]).head(10)
            cross.plot(kind="bar", ax=plt.gca(), colormap="Set2", edgecolor="black")
            plt.xlabel(x_col)
            plt.ylabel("العدد")
            plt.title(f"العلاقة بين {x_col} و {y_col}")
            plt.xticks(rotation=45, ha="right")

        plot_b64 = _fig_to_base64()
        ai_text = await analyze_relationship_arabic(df_work, x_col, y_col)
        return _render(request, df, plot=plot_b64, ai_analysis=ai_text)

    except Exception as e:
        plt.close()
        return _render(request, df, error=str(e))


@app.post("/heatmap", response_class=HTMLResponse)
async def heatmap(
    request: Request,
    session_id: Optional[str] = Cookie(default=None),
):
    df = _get_df(session_id)
    if df is None:
        return _render(request, None, error="لم يتم تحميل أي ملف")
    try:
        df_numeric = _auto_numeric(df)
        numeric_cols = df_numeric.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) < 2:
            return _render(request, df, error="لا توجد أعمدة رقمية كافية لرسم خريطة الحرارة")

        plt.figure(figsize=(10, 6))
        sns.heatmap(df_numeric[numeric_cols].corr(), annot=True, cmap="YlGnBu")

        plot_b64 = _fig_to_base64()
        ai_text = await analyze_heatmap_arabic(df_numeric)
        return _render(request, df, plot=plot_b64, ai_analysis=ai_text)
    except Exception as e:
        plt.close()
        return _render(request, df, error=str(e))


@app.post("/filter", response_class=HTMLResponse)
async def filter_data(
    request: Request,
    column: str = Form(...),
    value: str = Form(...),
    session_id: Optional[str] = Cookie(default=None),
):
    df = _get_df(session_id)
    if df is None:
        return _render(request, None, error="لم يتم تحميل أي ملف")
    try:
        filtered = df[df[column].astype(str).str.contains(value, case=False, na=False)]
        if filtered.empty:
            return _render(
                request, df,
                error=f"لا توجد نتائج للبحث عن '{value}' في عمود '{column}'",
            )
        return _render(request, filtered)
    except Exception as e:
        return _render(request, df, error=str(e))


@app.post("/clear", response_class=HTMLResponse)
async def clear_session(
    request: Request,
    session_id: Optional[str] = Cookie(default=None),
):
    if session_id and session_id in _sessions:
        del _sessions[session_id]
    response = _render(request, None)
    response.delete_cookie("session_id")
    return response

# ─── Chatbot ─────────────────────────────────────────────────────────────────

_chat_histories: dict[str, list] = {}


def _df_context(df: pd.DataFrame) -> str:
    """Build a COMPACT summary — keeps token count low for Groq free tier."""
    lines = []
    lines.append(f"الصفوف: {len(df)} | الأعمدة: {list(df.columns)}")

    # Numeric stats — only mean/min/max, rounded
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        lines.append("إحصائيات رقمية (المتوسط / الأدنى / الأعلى):")
        for col in num_cols[:8]:          # max 8 numeric cols
            s = df[col].dropna()
            if len(s):
                lines.append(f"  {col}: mean={s.mean():.2f}, min={s.min():.2f}, max={s.max():.2f}")

    # Categorical — top 3 values only
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    if cat_cols:
        lines.append("أعمدة نصية (أكثر القيم تكراراً):")
        for col in cat_cols[:5]:          # max 5 cat cols
            top = df[col].value_counts().head(3)
            lines.append(f"  {col}: {dict(top)}")

    # Only 5 sample rows
    lines.append("عينة (5 صفوف):")
    lines.append(df.head(5).to_string(index=False))

    return "\n".join(lines)


@app.post("/chat")
async def chat(
    request: Request,
    session_id: Optional[str] = Cookie(default=None),
):
    if not GROQ_API_KEY:
        return JSONResponse({"reply": "⚠️ لم يتم تعيين GROQ_API_KEY."})

    df = _get_df(session_id)
    if df is None:
        return JSONResponse({"reply": "⚠️ يرجى رفع ملف CSV أولاً."})

    body = await request.json()
    user_message = body.get("message", "").strip()
    if not user_message:
        return JSONResponse({"reply": ""})

    history = _chat_histories.get(session_id, [])

    system_prompt = f"""أنت محلل بيانات ذكي. أجب باللغة العربية بدقة وإيجاز.

ملخص البيانات:
{_df_context(df)}

- أجب فقط عما يخص هذه البيانات
- كن موجزاً ومحدداً"""

    history.append({"role": "user", "content": user_message})

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json={
                    "model": GROQ_MODEL,
                    "max_tokens": 512,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        *history[-6:],   # last 3 exchanges only
                    ],
                },
                timeout=30,
            )
            data = response.json()
            if response.status_code != 200:
                return JSONResponse({"reply": f"خطأ: {data.get('error', {}).get('message', str(data))}"})

            reply = data["choices"][0]["message"]["content"].strip()
            history.append({"role": "assistant", "content": reply})
            _chat_histories[session_id] = history[-10:]   # keep last 5 exchanges
            return JSONResponse({"reply": reply})

    except Exception as e:
        return JSONResponse({"reply": f"خطأ في الاتصال: {str(e)}"})


@app.post("/chat/clear")
async def clear_chat(
    request: Request,
    session_id: Optional[str] = Cookie(default=None),
):
    if session_id and session_id in _chat_histories:
        del _chat_histories[session_id]
    return JSONResponse({"status": "cleared"})


# ─── HR Report ────────────────────────────────────────────────────────────────

@app.post("/generate_report")
async def generate_report(request: Request, session_id: Optional[str] = Cookie(default=None)):
    df = _get_df(session_id)
    if df is None:
        return JSONResponse({"error": "لم يتم تحميل أي ملف"})
    try:
        quality  = data_quality_check(df)
        raw_kpis = calculate_kpis(df)
        analysis = await generate_smart_analysis(df, quality)
        report   = await generate_hr_report(raw_kpis, analysis, quality)
        return JSONResponse({"report": report})
    except Exception as e:
        return JSONResponse({"error": str(e)})


# ─── Chatbot ─────────────────────────────────────────────────────────────────

_chat_histories: dict[str, list] = {}


def _df_context(df: pd.DataFrame) -> str:
    lines = [f"الصفوف: {len(df)} | الأعمدة: {list(df.columns)}"]
    for col in df.select_dtypes(include="number").columns[:8]:
        s=df[col].dropna()
        if len(s): lines.append(f"  {col}: mean={s.mean():.2f} min={s.min():.2f} max={s.max():.2f}")
    for col in df.select_dtypes(exclude="number").columns[:5]:
        lines.append(f"  {col}: {dict(df[col].value_counts().head(3))}")
    lines.append(df.head(5).to_string(index=False))
    return "\n".join(lines)


@app.post("/chat")
async def chat(request: Request, session_id: Optional[str] = Cookie(default=None)):
    if not GROQ_API_KEY:
        return JSONResponse({"reply": "⚠️ لم يتم تعيين GROQ_API_KEY."})
    df = _get_df(session_id)
    if df is None:
        return JSONResponse({"reply": "⚠️ يرجى رفع ملف CSV أولاً."})
    body = await request.json()
    user_message = body.get("message", "").strip()
    if not user_message: return JSONResponse({"reply": ""})
    history = _chat_histories.get(session_id, [])
    history.append({"role": "user", "content": user_message})
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Content-Type":"application/json","Authorization":f"Bearer {GROQ_API_KEY}"},
                json={"model":GROQ_MODEL,"max_tokens":512,
                      "messages":[{"role":"system","content":f"أنت محلل بيانات ذكي. أجب بالعربية.\n{_df_context(df)}"},
                                  *history[-6:]]},
                timeout=30)
            data = resp.json()
            if resp.status_code != 200:
                return JSONResponse({"reply": f"خطأ: {data.get('error',{}).get('message',str(data))}"})
            reply = data["choices"][0]["message"]["content"].strip()
            history.append({"role":"assistant","content":reply})
            _chat_histories[session_id] = history[-10:]
            return JSONResponse({"reply": reply})
    except Exception as e:
        return JSONResponse({"reply": f"خطأ: {str(e)}"})


@app.post("/chat/clear")
async def clear_chat(request: Request, session_id: Optional[str] = Cookie(default=None)):
    if session_id and session_id in _chat_histories:
        del _chat_histories[session_id]
    return JSONResponse({"status": "cleared"})



# ══════════════════════════════════════════════════════════════════════
# PAYROLL ENGINE v2 — Professional HR Payroll System
# ══════════════════════════════════════════════════════════════════════

_payroll_sessions: dict[str, pd.DataFrame] = {}
_payroll_chat_histories: dict[str, list] = {}

WORKING_DAYS   = 22    # days/month
WORKING_HOURS  = 8     # hours/day
MAX_OT_HOURS   = 60    # warning threshold per month

DEFAULT_PAYROLL_RULES = {
    "regular_ot_rate":  1.5,
    "holiday_ot_rate":  2.0,
    "max_ot_hours":     MAX_OT_HOURS,
    "bonus_flat":       0.0,
}


# ─── Column Detection ─────────────────────────────────────────────────────────

def detect_payroll_columns(df: pd.DataFrame) -> dict:
    cols_lower = {c.lower().replace(" ","_").replace("-","_"): c for c in df.columns}

    def find(*names):
        for n in names:
            k = n.lower().replace(" ","_")
            if k in cols_lower: return cols_lower[k]
        return None

    return {k: v for k, v in {
        "id":          find("id","employee_id","emp_id","employeeid","employee_number","رقم_الموظف","الرقم_الوظيفي"),
        "name":        find("name","employee_name","emp_name","full_name","الاسم","اسم_الموظف"),
        "base_salary": find("base_salary","salary","basic_salary","monthlyincome","monthly_income","الراتب_الاساسي","الراتب"),
        "regular_ot":  find("regular_ot","regular_overtime","ot_hours","overtime_hours","overtime","اوفرتايم_عادي","ot_regular"),
        "holiday_ot":  find("holiday_ot","holiday_overtime","holiday_ot_hours","اوفرتايم_عطل","ot_holiday"),
        "social_sec":  find("social_security","social_insurance","ss_deduction","ضمان_اجتماعي","التامينات"),
        "bonus":       find("bonus","bonuses","incentive","بونص","مكافأة"),
        "other_ded":   find("other_deductions","deductions","other_ded","خصومات","خصومات_اخرى"),
    }.items() if v is not None}


# ─── Payroll Calculation (Pure Python — NO AI) ────────────────────────────────

def calculate_payroll_v2(df: pd.DataFrame, col_map: dict, rules: dict) -> list:
    results = []
    daily_rate_fn  = lambda base: base / WORKING_DAYS
    hourly_rate_fn = lambda base: base / WORKING_DAYS / WORKING_HOURS

    for idx, row in df.iterrows():
        emp = {}
        emp["idx"]  = int(idx)
        emp["id"]   = str(row[col_map["id"]])   if "id"   in col_map else str(idx + 1)
        emp["name"] = str(row[col_map["name"]]) if "name" in col_map else f"Employee {idx+1}"

        # Base salary
        base = 0.0
        if "base_salary" in col_map:
            try: base = float(row[col_map["base_salary"]])
            except: pass
        emp["base_salary"] = round(base, 3)

        hourly = hourly_rate_fn(base)

        # Regular overtime  (× 1.5)
        reg_ot_h = 0.0
        if "regular_ot" in col_map:
            try: reg_ot_h = float(row[col_map["regular_ot"]])
            except: pass
        emp["regular_ot_hours"] = round(reg_ot_h, 2)
        emp["regular_ot_pay"]   = round(hourly * rules["regular_ot_rate"] * reg_ot_h, 3)

        # Holiday overtime (× 2.0)
        hol_ot_h = 0.0
        if "holiday_ot" in col_map:
            try: hol_ot_h = float(row[col_map["holiday_ot"]])
            except: pass
        emp["holiday_ot_hours"] = round(hol_ot_h, 2)
        emp["holiday_ot_pay"]   = round(hourly * rules["holiday_ot_rate"] * hol_ot_h, 3)

        emp["total_ot_pay"] = round(emp["regular_ot_pay"] + emp["holiday_ot_pay"], 3)

        # Social security (entered as dinars directly)
        ss = 0.0
        if "social_sec" in col_map:
            try: ss = float(row[col_map["social_sec"]])
            except: pass
        emp["social_security"] = round(ss, 3)

        # Bonus
        bonus = rules.get("bonus_flat", 0.0)
        if "bonus" in col_map:
            try: bonus += float(row[col_map["bonus"]])
            except: pass
        emp["bonus"] = round(bonus, 3)

        # Other deductions
        other_ded = 0.0
        if "other_ded" in col_map:
            try: other_ded = float(row[col_map["other_ded"]])
            except: pass
        emp["other_deductions"] = round(other_ded, 3)

        emp["total_deductions"] = round(ss + other_ded, 3)

        # Final salary
        emp["final_salary"] = round(
            base + emp["regular_ot_pay"] + emp["holiday_ot_pay"]
            + emp["bonus"] - ss - other_ded,
            3
        )

        # Smart warnings
        emp["warnings"] = []
        total_ot = reg_ot_h + hol_ot_h
        if total_ot > rules.get("max_ot_hours", MAX_OT_HOURS):
            emp["warnings"].append(f"OT_HIGH:{total_ot:.0f}h")
        if base > 0 and emp["total_deductions"] > base * 0.5:
            emp["warnings"].append("DED_HIGH")
        if emp["final_salary"] < 0:
            emp["warnings"].append("NEG_SALARY")
        elif base > 0 and emp["final_salary"] < base * 0.4:
            emp["warnings"].append("LOW_SALARY")

        results.append(emp)
    return results


def payroll_summary_v2(results: list) -> dict:
    if not results: return {}
    finals  = [r["final_salary"]    for r in results]
    ot_pays = [r["total_ot_pay"]    for r in results]
    deds    = [r["total_deductions"] for r in results]
    return {
        "total_employees":   len(results),
        "total_payroll":     round(sum(finals), 3),
        "total_ot_cost":     round(sum(ot_pays), 3),
        "total_deductions":  round(sum(deds), 3),
        "avg_salary":        round(sum(finals)/len(finals), 3),
        "max_salary":        round(max(finals), 3),
        "min_salary":        round(min(finals), 3),
        "flagged":           sum(1 for r in results if r["warnings"]),
    }


async def ai_payroll_insights_v2(results: list, summary: dict, col_map: dict) -> dict:
    flagged = [r for r in results if r["warnings"]]
    dept_info = f"Available columns: {list(col_map.keys())}"

    prompt = f"""You are a senior HR operations and payroll analytics expert.
You are given pre-calculated payroll results. Do NOT recalculate anything.
Detect operational issues, risks, anomalies. Write everything in Arabic.

PAYROLL SUMMARY:
- Total Employees: {summary['total_employees']}
- Total Payroll: {summary['total_payroll']:,.3f}
- Total OT Cost: {summary['total_ot_cost']:,.3f}
- Total Deductions: {summary['total_deductions']:,.3f}
- Avg Salary: {summary['avg_salary']:,.3f}
- Max Salary: {summary['max_salary']:,.3f}
- Min Salary: {summary['min_salary']:,.3f}
- Flagged: {summary['flagged']}

FLAGGED EMPLOYEES ({len(flagged)}):
{chr(10).join([f"- {r['name']}: {', '.join(r['warnings'])}" for r in flagged[:15]])}

{dept_info}

Return ONLY valid JSON (no markdown):
{{
  "insights": ["ملاحظة تشغيلية محددة 1","ملاحظة تشغيلية محددة 2","ملاحظة تشغيلية محددة 3"],
  "top_risks": ["خطر مؤثر مع السبب والتأثير 1","خطر مؤثر مع السبب والتأثير 2"],
  "recommendations": ["إجراء محدد قابل للتنفيذ 1","إجراء محدد قابل للتنفيذ 2","إجراء محدد قابل للتنفيذ 3"]
}}"""

    raw = await ai_analyze_arabic(prompt)
    try:
        cleaned = raw.strip()
        if "```" in cleaned:
            for p in cleaned.split("```"):
                p = p.strip()
                if p.startswith("json"): p = p[4:].strip()
                if p.startswith("{"): cleaned = p; break
        s = cleaned.find("{"); e = cleaned.rfind("}")+1
        if s != -1 and e > s: cleaned = cleaned[s:e]
        return json.loads(cleaned)
    except:
        return {"insights": [], "top_risks": [], "recommendations": []}


# ─── Payroll Routes ───────────────────────────────────────────────────────────

@app.get("/payroll", response_class=HTMLResponse)
async def payroll_page(request: Request):
    return templates.TemplateResponse(request=request, name="payroll.html", context={
        "rules": DEFAULT_PAYROLL_RULES,
        "col_map": None, "results": None, "summary": None,
        "insights": None, "error": None, "columns": None
    })


@app.post("/payroll/upload", response_class=HTMLResponse)
async def payroll_upload(request: Request, file: UploadFile = File(...)):
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("الملف كبير جداً — الحد الأقصى 10MB")
        fname = file.filename.lower()
        if fname.endswith(".csv"):
            try: df = pd.read_csv(io.BytesIO(content))
            except: df = pd.read_csv(io.BytesIO(content), encoding="ISO-8859-1")
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError("يرجى رفع ملف CSV أو Excel فقط")
        if df.empty: raise ValueError("الملف فارغ")

        sid = str(uuid.uuid4())
        _payroll_sessions[sid] = df
        col_map = detect_payroll_columns(df)

        # Auto-calculate on upload
        results = calculate_payroll_v2(df, col_map, DEFAULT_PAYROLL_RULES)
        summary = payroll_summary_v2(results)
        insights = await ai_payroll_insights_v2(results, summary, col_map)

        response = templates.TemplateResponse(request=request, name="payroll.html", context={
            "rules": DEFAULT_PAYROLL_RULES,
            "col_map": col_map,
            "columns": list(df.columns),
            "results": results,
            "summary": summary,
            "insights": insights,
            "error": None
        })
        response.set_cookie("payroll_sid", sid)
        return response
    except Exception as e:
        return templates.TemplateResponse(request=request, name="payroll.html", context={
            "rules": DEFAULT_PAYROLL_RULES,
            "col_map": None, "results": None, "summary": None,
            "insights": None, "error": str(e), "columns": None
        })


@app.post("/payroll/recalculate")
async def payroll_recalculate(request: Request, payroll_sid: Optional[str] = Cookie(default=None)):
    """Live recalculate after inline edit — returns JSON."""
    df = _payroll_sessions.get(payroll_sid)
    if df is None:
        return JSONResponse({"error": "session expired"}, status_code=400)
    try:
        body = await request.json()
        edits  = body.get("edits", {})   # {idx: {field: value}}
        rules  = body.get("rules", DEFAULT_PAYROLL_RULES)
        col_map = detect_payroll_columns(df)

        results = calculate_payroll_v2(df, col_map, rules)

        # Apply inline edits on top of calculated results
        for idx_str, fields in edits.items():
            idx = int(idx_str)
            for r in results:
                if r["idx"] == idx:
                    for field, val in fields.items():
                        try: r[field] = float(val)
                        except: pass
                    # Recalculate final salary with edited values
                    base  = r["base_salary"]
                    hourly = base / WORKING_DAYS / WORKING_HOURS
                    r["regular_ot_pay"] = round(hourly * rules.get("regular_ot_rate",1.5) * r["regular_ot_hours"], 3)
                    r["holiday_ot_pay"] = round(hourly * rules.get("holiday_ot_rate",2.0) * r["holiday_ot_hours"], 3)
                    r["total_ot_pay"]   = round(r["regular_ot_pay"] + r["holiday_ot_pay"], 3)
                    r["total_deductions"]= round(r["social_security"] + r["other_deductions"], 3)
                    r["final_salary"]   = round(
                        base + r["regular_ot_pay"] + r["holiday_ot_pay"]
                        + r["bonus"] - r["social_security"] - r["other_deductions"], 3
                    )
                    break

        summary = payroll_summary_v2(results)
        return JSONResponse({"results": results, "summary": summary})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/payroll/chat")
async def payroll_chat(request: Request, payroll_sid: Optional[str] = Cookie(default=None)):
    if not GROQ_API_KEY:
        return JSONResponse({"reply": "⚠️ لم يتم تعيين GROQ_API_KEY."})
    df = _payroll_sessions.get(payroll_sid)
    body = await request.json()
    user_message = body.get("message", "").strip()
    if not user_message: return JSONResponse({"reply": ""})

    history = _payroll_chat_histories.get(payroll_sid, [])

    context = ""
    if df is not None:
        col_map = detect_payroll_columns(df)
        results = calculate_payroll_v2(df, col_map, DEFAULT_PAYROLL_RULES)
        summary = payroll_summary_v2(results)
        context = f"""بيانات الرواتب:
- الموظفون: {summary['total_employees']}
- إجمالي الرواتب: {summary['total_payroll']:,.3f}
- إجمالي الأوفرتايم: {summary['total_ot_cost']:,.3f}
- إجمالي الخصومات: {summary['total_deductions']:,.3f}
- أعلى راتب: {summary['max_salary']:,.3f}
- أقل راتب: {summary['min_salary']:,.3f}
- موظفون مُبلَّغ عنهم: {summary['flagged']}
الأعمدة المتوفرة: {list(col_map.keys())}"""

    system = f"""أنت خبير رواتب وموارد بشرية محترف. أجب بالعربية بدقة وإيجاز.
لا تعيد حساب أي قيمة — استخدم فقط البيانات المعطاة.
{context}"""

    history.append({"role": "user", "content": user_message})
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Content-Type":"application/json","Authorization":f"Bearer {GROQ_API_KEY}"},
                json={"model":GROQ_MODEL,"max_tokens":512,
                      "messages":[{"role":"system","content":system},*history[-6:]]},
                timeout=30)
            data = resp.json()
            if resp.status_code != 200:
                return JSONResponse({"reply": f"خطأ: {data.get('error',{}).get('message','')}"})
            reply = data["choices"][0]["message"]["content"].strip()
            history.append({"role":"assistant","content":reply})
            _payroll_chat_histories[payroll_sid] = history[-12:]
            return JSONResponse({"reply": reply})
    except Exception as e:
        return JSONResponse({"reply": f"خطأ: {str(e)}"})


@app.post("/payroll/chat/clear")
async def payroll_chat_clear(request: Request, payroll_sid: Optional[str] = Cookie(default=None)):
    if payroll_sid and payroll_sid in _payroll_chat_histories:
        del _payroll_chat_histories[payroll_sid]
    return JSONResponse({"status": "cleared"})


@app.post("/payroll/clear", response_class=HTMLResponse)
async def payroll_clear(request: Request, payroll_sid: Optional[str] = Cookie(default=None)):
    if payroll_sid and payroll_sid in _payroll_sessions:
        del _payroll_sessions[payroll_sid]
    response = templates.TemplateResponse(request=request, name="payroll.html", context={
        "rules": DEFAULT_PAYROLL_RULES,
        "col_map": None, "results": None, "summary": None,
        "insights": None, "error": None, "columns": None
    })
    response.delete_cookie("payroll_sid")
    return response