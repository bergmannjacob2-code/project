"""
Advanced Streamlit DCF Valuation App
FINA 4011/5011 Project 2

How to run locally:
1. Install packages:
   pip install streamlit pandas numpy xlsxwriter yfinance
2. Run:
   streamlit run advanced_dcf_valuation_app.py

For Streamlit Cloud, create a requirements.txt with:
streamlit
pandas
numpy
xlsxwriter
yfinance
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Advanced DCF Valuation App",
    page_icon="📊",
    layout="wide",
)


# -----------------------------
# Helper functions
# -----------------------------
def safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def dollars(x):
    return f"${x:,.2f}"


def dollars_millions(x):
    return f"${x:,.2f} M"


def pct(x):
    return f"{x * 100:,.2f}%"


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(ticker):
    """Fetch key market and financial inputs from yfinance.

    The app still lets users manually override every value in the sidebar
    because market data can be missing or delayed for some tickers.
    """
    clean_ticker = ticker.upper().strip() if ticker else ""
    fallback = {
        "company_name": clean_ticker or "Manual Company",
        "current_price": None,
        "shares_outstanding": None,
        "market_cap": None,
        "total_debt": 0,
        "cash": 0,
        "revenue": None,
        "profit_margin": None,
        "beta": None,
        "sector": "Unavailable",
        "industry": "Unavailable",
        "data_source": "Manual fallback",
    }
    if yf is None:
        return fallback, "yfinance is not installed. Add yfinance to requirements.txt or install it locally, then rerun the app."
    if not clean_ticker:
        return fallback, "Enter a ticker symbol, then click Refresh market data."
    try:
        stock = yf.Ticker(clean_ticker)
        try:
            fast = stock.fast_info
        except Exception:
            fast = {}
        try:
            info = stock.info or {}
        except Exception:
            info = {}

        current_price = (
            safe_float(getattr(fast, "last_price", None), None)
            or safe_float(info.get("currentPrice"), None)
            or safe_float(info.get("regularMarketPrice"), None)
            or safe_float(info.get("previousClose"), None)
        )
        shares_outstanding = (
            safe_float(getattr(fast, "shares", None), None)
            or safe_float(info.get("sharesOutstanding"), None)
        )
        market_cap = (
            safe_float(getattr(fast, "market_cap", None), None)
            or safe_float(info.get("marketCap"), None)
        )

        revenue = safe_float(info.get("totalRevenue"), None)
        try:
            financials = stock.financials
            if revenue is None and financials is not None and not financials.empty and "Total Revenue" in financials.index:
                revenue = safe_float(financials.loc["Total Revenue"].iloc[0], None)
            if revenue is None and financials is not None and not financials.empty and "Operating Revenue" in financials.index:
                revenue = safe_float(financials.loc["Operating Revenue"].iloc[0], None)
        except Exception:
            pass

        data = {
            "company_name": info.get("longName") or info.get("shortName") or clean_ticker,
            "current_price": current_price,
            "shares_outstanding": shares_outstanding,
            "market_cap": market_cap,
            "total_debt": safe_float(info.get("totalDebt"), 0),
            "cash": safe_float(info.get("totalCash"), 0),
            "revenue": revenue,
            "profit_margin": safe_float(info.get("profitMargins"), None),
            "beta": safe_float(info.get("beta"), None),
            "sector": info.get("sector", "Unavailable"),
            "industry": info.get("industry", "Unavailable"),
            "data_source": "yfinance",
        }
        missing = [label for label, value in {"price": data["current_price"], "shares": data["shares_outstanding"], "revenue": data["revenue"]}.items() if value is None]
        if missing:
            note = f"Market data loaded from yfinance, but these fields were missing: {', '.join(missing)}. Use the sidebar overrides for missing values."
        else:
            note = "Market data loaded from yfinance. You can still override all values in the sidebar."
        return data, note
    except Exception as exc:
        fallback["company_name"] = clean_ticker
        return fallback, f"Unable to fetch yfinance data for {clean_ticker}. Use manual sidebar inputs. Error: {exc}"

def build_dcf_projection(
    current_revenue_m,
    years,
    growth_rates,
    ebit_margin,
    tax_rate,
    depreciation_pct_revenue,
    capex_pct_revenue,
    nwc_pct_revenue,
    wacc,
    terminal_growth,
    debt_m,
    cash_m,
    shares_m,
):
    rows = []
    previous_revenue = current_revenue_m

    for year in range(1, years + 1):
        growth = growth_rates[year - 1]
        revenue = previous_revenue * (1 + growth)
        revenue_change = revenue - previous_revenue
        ebit = revenue * ebit_margin
        taxes = ebit * tax_rate
        nopat = ebit - taxes
        depreciation = revenue * depreciation_pct_revenue
        capex = revenue * capex_pct_revenue
        change_nwc = revenue_change * nwc_pct_revenue
        fcf = nopat + depreciation - capex - change_nwc
        discount_factor = 1 / ((1 + wacc) ** year)
        pv_fcf = fcf * discount_factor

        rows.append({
            "Year": year,
            "Revenue Growth": growth,
            "Revenue": revenue,
            "EBIT": ebit,
            "Taxes": taxes,
            "NOPAT": nopat,
            "Depreciation": depreciation,
            "CapEx": capex,
            "Change in NWC": change_nwc,
            "Free Cash Flow": fcf,
            "Discount Factor": discount_factor,
            "PV of FCF": pv_fcf,
        })
        previous_revenue = revenue

    df = pd.DataFrame(rows)
    final_fcf = df["Free Cash Flow"].iloc[-1]

    if wacc <= terminal_growth:
        raise ValueError("WACC must be greater than terminal growth rate for the Gordon Growth terminal value formula.")

    terminal_value = final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / ((1 + wacc) ** years)
    enterprise_value = df["PV of FCF"].sum() + pv_terminal_value
    equity_value = enterprise_value - debt_m + cash_m
    value_per_share = equity_value / shares_m if shares_m > 0 else np.nan

    summary = {
        "Terminal Value": terminal_value,
        "PV of Terminal Value": pv_terminal_value,
        "PV of Projected FCF": df["PV of FCF"].sum(),
        "Enterprise Value": enterprise_value,
        "Debt": debt_m,
        "Cash": cash_m,
        "Equity Value": equity_value,
        "Intrinsic Value per Share": value_per_share,
    }
    return df, summary


def create_excel_replication(dcf_df, summary_df, assumptions_df):
    """Return an Excel file as bytes for easy validation submission."""
    from io import BytesIO

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        assumptions_df.to_excel(writer, sheet_name="Assumptions", index=False)
        dcf_df.to_excel(writer, sheet_name="DCF Projection", index=False)
        summary_df.to_excel(writer, sheet_name="Valuation Output", index=False)

        workbook = writer.book
        money_fmt = workbook.add_format({"num_format": "$#,##0.00"})
        pct_fmt = workbook.add_format({"num_format": "0.00%"})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})

        for sheet_name in ["Assumptions", "DCF Projection", "Valuation Output"]:
            ws = writer.sheets[sheet_name]
            ws.set_row(0, None, header_fmt)
            ws.set_column(0, 15, 18)

        ws = writer.sheets["DCF Projection"]
        pct_columns = [1, 10]
        money_columns = list(range(2, 10)) + [11]
        for col in pct_columns:
            ws.set_column(col, col, 16, pct_fmt)
        for col in money_columns:
            ws.set_column(col, col, 18, money_fmt)

        writer.sheets["Valuation Output"].set_column(1, 1, 20, money_fmt)
    return output.getvalue()


# -----------------------------
# Main title and instructions
# -----------------------------
st.title("📊 Advanced DCF Equity Valuation App")
st.caption("Built for a finance project: estimates a stock's intrinsic value using a discounted cash flow model and compares it to market price.")

with st.expander("How to use this app", expanded=True):
    st.markdown(
        """
        1. Enter a stock ticker or company name.  
        2. Enter the company data and assumptions in the sidebar manually.  
        3. Read the valuation summary, projection table, charts, and sensitivity analysis.  
        4. Download the Excel replication file to show that the app's output can be checked manually.
        """
    )


# -----------------------------
# Ticker and yfinance market data
# -----------------------------
left, right = st.columns([1, 2])
with left:
    ticker = st.text_input("Stock ticker", value="AAPL", help="Examples: AAPL, MSFT, TSLA, NVDA").upper().strip()
    load_data = st.button("Refresh market data")

if "market_data" not in st.session_state:
    st.session_state.market_data, st.session_state.market_note = get_stock_data("AAPL")

if load_data and ticker:
    with st.spinner(f"Loading market data for {ticker}..."):
        st.session_state.market_data, st.session_state.market_note = get_stock_data(ticker)

market_data = st.session_state.market_data

with right:
    if market_data.get("data_source") == "yfinance":
        st.success(st.session_state.market_note)
    else:
        st.warning(st.session_state.market_note)

company_name = market_data.get("company_name", ticker)
st.subheader(f"Company: {company_name}")

market_cols = st.columns(5)
current_price_raw = market_data.get("current_price")
market_cap_raw = market_data.get("market_cap")
shares_raw = market_data.get("shares_outstanding")
revenue_raw = market_data.get("revenue")
beta_raw = market_data.get("beta")

market_cols[0].metric("Market Price", "Manual Input" if current_price_raw is None else dollars(current_price_raw))
market_cols[1].metric("Market Cap", "Manual Input" if market_cap_raw is None else dollars(market_cap_raw))
market_cols[2].metric("Shares", "Manual Input" if shares_raw is None else f"{shares_raw/1_000_000:,.2f} M")
market_cols[3].metric("Revenue", "Manual Input" if revenue_raw is None else dollars(revenue_raw))
market_cols[4].metric("Beta", "Manual Input" if beta_raw is None else f"{beta_raw:,.2f}")

st.write(f"**Data source:** {market_data.get('data_source', 'Manual fallback')}")
st.write(f"**Sector:** {market_data.get('sector', 'Unavailable')} | **Industry:** {market_data.get('industry', 'Unavailable')}")
with st.expander("Retrieved market data and manual override note"):
    st.write("The app pulls available values from yfinance, then places them into editable sidebar inputs. If a field is missing or looks unreasonable, type your own value in the sidebar.")
    st.json({
        "Ticker": ticker,
        "Company": company_name,
        "Current Price": current_price_raw,
        "Market Cap": market_cap_raw,
        "Shares Outstanding": shares_raw,
        "Revenue": revenue_raw,
        "Total Debt": market_data.get("total_debt"),
        "Cash": market_data.get("cash"),
        "Profit Margin": market_data.get("profit_margin"),
        "Beta": beta_raw,
    })


# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.title("Valuation Assumptions")
st.sidebar.caption("Amounts are entered in millions unless marked otherwise.")

st.sidebar.header("Company Data")
default_revenue_m = revenue_raw / 1_000_000 if revenue_raw else 1000.0
default_debt_m = market_data.get("total_debt", 0) / 1_000_000 if market_data.get("total_debt") else 0.0
default_cash_m = market_data.get("cash", 0) / 1_000_000 if market_data.get("cash") else 0.0
default_shares_m = shares_raw / 1_000_000 if shares_raw else 100.0
default_price = current_price_raw if current_price_raw else 50.0

current_revenue_m = st.sidebar.number_input("Current Revenue ($ millions)", min_value=0.01, value=float(default_revenue_m), step=100.0)
debt_m = st.sidebar.number_input("Debt ($ millions)", min_value=0.0, value=float(default_debt_m), step=100.0)
cash_m = st.sidebar.number_input("Cash & Equivalents ($ millions)", min_value=0.0, value=float(default_cash_m), step=100.0)
shares_m = st.sidebar.number_input("Shares Outstanding (millions)", min_value=0.01, value=float(default_shares_m), step=10.0)
current_market_price = st.sidebar.number_input("Current Market Price per Share ($)", min_value=0.01, value=float(default_price), step=1.0)

st.sidebar.header("Forecast Settings")
years = st.sidebar.slider("Projection Years", min_value=3, max_value=10, value=5)
growth_model = st.sidebar.radio("Revenue Growth Method", ["Same growth every year", "Fade from high growth to stable growth", "Custom year-by-year growth"])

if growth_model == "Same growth every year":
    growth = st.sidebar.number_input("Annual Revenue Growth (%)", value=6.0, step=0.5) / 100
    growth_rates = [growth] * years
elif growth_model == "Fade from high growth to stable growth":
    high_growth = st.sidebar.number_input("Year 1 Revenue Growth (%)", value=10.0, step=0.5) / 100
    stable_growth = st.sidebar.number_input("Final Projection Year Growth (%)", value=3.0, step=0.5) / 100
    growth_rates = list(np.linspace(high_growth, stable_growth, years))
else:
    growth_rates = []
    for i in range(1, years + 1):
        rate = st.sidebar.number_input(f"Year {i} Growth (%)", value=6.0, step=0.5, key=f"custom_growth_{i}") / 100
        growth_rates.append(rate)

st.sidebar.header("Operating Assumptions")
default_margin = market_data.get("profit_margin")
if default_margin is None or default_margin <= 0:
    default_margin = 0.20

ebit_margin = st.sidebar.number_input("EBIT Margin (%)", value=float(default_margin * 100), step=0.5, help="EBIT margin estimates operating profit as a percentage of revenue.") / 100
tax_rate = st.sidebar.number_input("Tax Rate (%)", value=21.0, step=0.5, help="Use the company's effective tax rate or a reasonable corporate tax assumption.") / 100
depreciation_pct_revenue = st.sidebar.number_input("Depreciation as % of Revenue", value=3.0, step=0.25) / 100
capex_pct_revenue = st.sidebar.number_input("Capital Expenditures as % of Revenue", value=4.0, step=0.25) / 100
nwc_pct_revenue = st.sidebar.number_input("Net Working Capital Investment as % of Revenue Growth", value=10.0, step=0.5, help="This applies only to the change in revenue, not total revenue.") / 100

st.sidebar.header("Discount Rate and Terminal Value")
wacc = st.sidebar.number_input("WACC / Discount Rate (%)", value=9.0, step=0.25, help="Weighted Average Cost of Capital. Higher WACC lowers valuation.") / 100
terminal_growth = st.sidebar.number_input("Terminal Growth Rate (%)", value=2.5, step=0.25, help="Long-run perpetual FCF growth. Usually should be below WACC.") / 100

st.sidebar.subheader("Optional WACC Helper")
use_wacc_helper = st.sidebar.checkbox("Estimate WACC using beta and capital structure", value=False)
if use_wacc_helper:
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.25, step=0.25) / 100
    market_return = st.sidebar.number_input("Expected Market Return (%)", value=9.50, step=0.25) / 100
    pretax_cost_debt = st.sidebar.number_input("Pre-Tax Cost of Debt (%)", value=5.50, step=0.25) / 100
    beta_for_wacc = st.sidebar.number_input("Beta", value=float(beta_raw) if beta_raw else 1.00, step=0.05)

    market_equity_m = current_market_price * shares_m
    total_capital_m = market_equity_m + debt_m
    if total_capital_m > 0:
        weight_equity = market_equity_m / total_capital_m
        weight_debt = debt_m / total_capital_m
        cost_equity = risk_free_rate + beta_for_wacc * (market_return - risk_free_rate)
        after_tax_cost_debt = pretax_cost_debt * (1 - tax_rate)
        estimated_wacc = (weight_equity * cost_equity) + (weight_debt * after_tax_cost_debt)
        st.sidebar.info(f"Estimated WACC: {estimated_wacc * 100:.2f}%")
        wacc = estimated_wacc


# -----------------------------
# DCF calculation
# -----------------------------
try:
    dcf_df, summary = build_dcf_projection(
        current_revenue_m=current_revenue_m,
        years=years,
        growth_rates=growth_rates,
        ebit_margin=ebit_margin,
        tax_rate=tax_rate,
        depreciation_pct_revenue=depreciation_pct_revenue,
        capex_pct_revenue=capex_pct_revenue,
        nwc_pct_revenue=nwc_pct_revenue,
        wacc=wacc,
        terminal_growth=terminal_growth,
        debt_m=debt_m,
        cash_m=cash_m,
        shares_m=shares_m,
    )

    intrinsic_value = summary["Intrinsic Value per Share"]
    upside_downside = (intrinsic_value / current_market_price) - 1

    # -----------------------------
    # Results
    # -----------------------------
    st.header("1. Valuation Summary")
    result_cols = st.columns(4)
    result_cols[0].metric("Intrinsic Value / Share", dollars(intrinsic_value))
    result_cols[1].metric("Current Market Price", dollars(current_market_price))
    result_cols[2].metric("Implied Upside / Downside", pct(upside_downside))
    result_cols[3].metric("Conclusion", "Undervalued" if upside_downside > 0 else "Overvalued")

    if upside_downside > 0.15:
        st.success("Based on these assumptions, the stock appears undervalued by more than 15%.")
    elif upside_downside < -0.15:
        st.error("Based on these assumptions, the stock appears overvalued by more than 15%.")
    else:
        st.warning("Based on these assumptions, the stock appears close to fair value.")

    st.header("2. Step-by-Step DCF Breakdown")
    st.markdown(
        """
        **DCF logic used by this app:**  
        Revenue is projected forward using the selected growth assumptions. EBIT is estimated from the EBIT margin. Taxes are subtracted to calculate NOPAT. Free Cash Flow is then calculated as:
        
        **FCF = NOPAT + Depreciation - Capital Expenditures - Change in Net Working Capital**
        
        Each year's FCF is discounted back to today using WACC. A terminal value is calculated at the end of the projection period using the Gordon Growth Method:
        
        **Terminal Value = Final Year FCF × (1 + Terminal Growth) / (WACC - Terminal Growth)**
        
        Enterprise Value equals the present value of projected FCF plus the present value of terminal value. Equity Value equals Enterprise Value minus debt plus cash. Intrinsic value per share equals Equity Value divided by shares outstanding.
        """
    )

    summary_df = pd.DataFrame({
        "Item": list(summary.keys()),
        "Value ($ millions, except per-share)": list(summary.values()),
    })

    st.subheader("Valuation Output")
    display_summary = summary_df.copy()
    display_summary["Value ($ millions, except per-share)"] = display_summary["Value ($ millions, except per-share)"].map(lambda x: f"{x:,.2f}")
    st.dataframe(display_summary, use_container_width=True)

    st.header("3. Projection Table")
    display_df = dcf_df.copy()
    for col in ["Revenue Growth", "Discount Factor"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:,.2%}" if col == "Revenue Growth" else f"{x:,.4f}")
    money_cols = ["Revenue", "EBIT", "Taxes", "NOPAT", "Depreciation", "CapEx", "Change in NWC", "Free Cash Flow", "PV of FCF"]
    for col in money_cols:
        display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}")
    st.dataframe(display_df, use_container_width=True)

    st.header("4. Charts")
    chart_col1, chart_col2 = st.columns(2)

    chart_df = dcf_df.set_index("Year")

    with chart_col1:
        st.subheader("Revenue Projection")
        st.line_chart(chart_df[["Revenue"]])
        st.caption("Revenue is shown in $ millions.")

    with chart_col2:
        st.subheader("Free Cash Flow Projection")
        st.bar_chart(chart_df[["Free Cash Flow"]])
        st.caption("Free cash flow is shown in $ millions.")

    st.header("5. Sensitivity Analysis")
    st.write("This table shows how intrinsic value per share changes when WACC and terminal growth assumptions change.")

    wacc_range = np.array([wacc - 0.02, wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02])
    tg_range = np.array([terminal_growth - 0.01, terminal_growth - 0.005, terminal_growth, terminal_growth + 0.005, terminal_growth + 0.01])
    wacc_range = np.maximum(wacc_range, 0.001)
    tg_range = np.maximum(tg_range, -0.05)

    sensitivity = pd.DataFrame(index=[pct(x) for x in tg_range], columns=[pct(x) for x in wacc_range])
    for tg in tg_range:
        for wa in wacc_range:
            if wa <= tg:
                sensitivity.loc[pct(tg), pct(wa)] = "N/A"
            else:
                _, temp_summary = build_dcf_projection(
                    current_revenue_m, years, growth_rates, ebit_margin, tax_rate,
                    depreciation_pct_revenue, capex_pct_revenue, nwc_pct_revenue,
                    wa, tg, debt_m, cash_m, shares_m
                )
                sensitivity.loc[pct(tg), pct(wa)] = f"${temp_summary['Intrinsic Value per Share']:,.2f}"

    st.dataframe(sensitivity, use_container_width=True)
    st.caption("Rows = terminal growth rate. Columns = WACC.")

    st.header("6. Assumptions Used")
    assumptions = {
        "Ticker": ticker,
        "Company": company_name,
        "Current Revenue ($M)": current_revenue_m,
        "Projection Years": years,
        "Growth Rates": ", ".join([pct(x) for x in growth_rates]),
        "EBIT Margin": pct(ebit_margin),
        "Tax Rate": pct(tax_rate),
        "Depreciation / Revenue": pct(depreciation_pct_revenue),
        "CapEx / Revenue": pct(capex_pct_revenue),
        "NWC Investment / Revenue Growth": pct(nwc_pct_revenue),
        "WACC": pct(wacc),
        "Terminal Growth": pct(terminal_growth),
        "Debt ($M)": debt_m,
        "Cash ($M)": cash_m,
        "Shares Outstanding (M)": shares_m,
        "Current Market Price": current_market_price,
        "Data Source": market_data.get("data_source", "Manual fallback"),
        "Date Generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    assumptions_df = pd.DataFrame(list(assumptions.items()), columns=["Assumption", "Value"])
    st.dataframe(assumptions_df, use_container_width=True)

    st.header("7. Download for Submission / Excel Replication")
    st.write("Download this file to help complete the validation Excel requirement. It includes assumptions, the DCF projection, and valuation output.")

    excel_bytes = create_excel_replication(dcf_df, summary_df, assumptions_df)
    st.download_button(
        label="Download Excel Replication File",
        data=excel_bytes,
        file_name=f"{ticker}_DCF_replication.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    csv = dcf_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Projection Table as CSV",
        data=csv,
        file_name=f"{ticker}_dcf_projection.csv",
        mime="text/csv",
    )

except ValueError as exc:
    st.error(str(exc))
    st.write("Fix the assumption issue in the sidebar and rerun the model.")
except Exception as exc:
    st.error("Something went wrong while calculating the DCF model.")
    st.write(exc)


# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Educational model only. This app is not financial advice. Results depend heavily on assumptions.")
