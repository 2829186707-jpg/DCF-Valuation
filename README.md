# 📊 DCF 智能估值面板

输入股票代码 → 自动抓取财务数据 → **DCF / DDM / 反向DCF / 可比公司** 四种方法交叉估值 → 综合研判。

支持 **A股** 和 **美股**，无需任何 API Key 即可使用（财务数据来自公开免费数据源）。

---

## ✨ 功能特性

| 特性 | 说明 |
|---|---|
| 🎯 一键估值 | 输入股票代码或名称（A股如 `600519`/`贵州茅台`，美股如 `AAPL`/`Apple`），自动完成全流程 |
| 🏷️ 估值风格 | 按公司属性（**成长/稳健/价值/周期股**，可自动识别或手动指定）套用对应增长率、永续增长与风险溢价参数 |
| 🎯 推荐合理估值 | 综合研判页汇总各方法中位数，给出**推荐合理估值 + 合理区间 + 对应合理市值**与综合结论 |
| 💰 四种估值方法 | DCF（FCFF 两阶段）、DDM 股利贴现、反向 DCF、可比公司法 |
| 📖 方法论讲解 | 每个方法都内置公式、原理、用法、适用场景与局限说明 |
| ⚙️ 假设可调 | 所有预测假设（增长率、WACC、利润率等）自动生成 + 可手动调整 |
| 📈 可视化 | FCFF 预测图、价值瀑布图、敏感性热力图、多方法对比图 |
| 🤖 AI 综合研判 | 可选接入大模型（OpenAI/DeepSeek/豆包/通义/Kimi），无 Key 时用本地规则引擎兜底 |
| 🌍 多市场 | A股（新浪/腾讯数据）、美股（SEC 官方财务 + 腾讯/新浪行情） |

## 🗂️ 数据源（全部免费、无需 key）

| 数据 | A股 | 美股 |
|---|---|---|
| 行情/价格/市值/PE/PB | 腾讯行情接口 | 腾讯行情接口 |
| 历史日线（Beta） | 新浪日线 | 新浪美股日线 |
| 财务三表 | 新浪财务报表 | **SEC EDGAR 官方 API**（备用：Yahoo Finance） |
| 无风险利率 | 中债 10 年期 | 美债 10 年期 |
| 指数（Beta 基准） | 沪深 300 | 标普 500 (.INX) |

## 🚀 本地运行

```bash
# 1. 创建虚拟环境（可选但推荐）
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。

## ☁️ 部署到 GitHub + Streamlit Cloud（免费）

1. **推送到 GitHub**（示例）：
   ```bash
   git init
   git add .
   git commit -m "init DCF valuation panel"
   git branch -M main
   git remote add origin https://github.com/<你的用户名>/<仓库名>.git
   git push -u origin main
   ```

2. **注册/登录** [Streamlit Community Cloud](https://streamlit.io/cloud)（用 GitHub 账号登录）

3. **Deploy**：
   - 点击 **New app**
   - 选择仓库 + 分支 `main` + 主文件 `app.py`
   - 点击 **Deploy**

4. 部署完成后会获得一个公网链接（形如 `https://<你的用户名>-<仓库名>.streamlit.app`），像网页一样直接访问使用。

> 💡 部署后即开即用，无需配置任何环境变量/密钥。AI 接口的 API Key 由使用者在**界面内自行填写**（仅存于浏览器会话，不落盘、不上传）。

## 📁 项目结构

```
DCF/
├── app.py                    # Streamlit 主应用（入口）
├── requirements.txt          # 依赖
├── .streamlit/config.toml    # 主题配置
├── src/
│   ├── data_fetcher.py       # 数据抓取层（A股/美股统一化）
│   ├── sec_fetcher.py        # SEC EDGAR 美股财务解析
│   ├── wacc.py               # WACC 计算（CAPM）
│   ├── methodology.py        # 四种方法的方法论讲解
│   ├── ai_advisor.py         # AI 综合研判（多服务商 + 规则引擎）
│   └── models/
│       ├── dcf.py            # DCF 现金流折现（FCFF 两阶段）
│       ├── ddm.py            # DDM 股利贴现
│       ├── reverse_dcf.py    # 反向 DCF
│       └── comps.py          # 可比公司法
```

## 📖 估值方法速览

- **DCF**：`FCFF = EBIT×(1−t) + 折旧摊销 − 资本开支 − Δ营运资本`，两阶段折现，适用于现金流稳定的公司
- **DDM**：未来股利现值之和，适用于稳定分红公司（银行/公用事业/消费龙头）
- **反向 DCF**：反推当前股价隐含的增长率，与历史对比揭示预期差
- **可比公司**：同业中位 PE/PB/EV-EBITDA 套用，相对估值交叉验证

## ⚠️ 免责声明

本工具基于公开财务数据自动计算，仅供**研究学习参考**，**不构成任何投资建议**。数据可能滞后或有误差，重要投资决策请以公司公告和独立尽职调查为准。
