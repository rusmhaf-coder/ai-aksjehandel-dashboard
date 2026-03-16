import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

START_CAPITAL_NOK = 20_000
RISK_PER_TRADE = 0.01
MAX_POSITION_PCT = 0.20
MIN_SCORE_BUY = 7.2
MIN_SCORE_WATCH = 5.8
MIN_RR = 1.8
LOOKBACK_PERIOD = "18mo"
TOP_N = 15

OSLO_TICKERS = [
    "EQNR.OL", "DNB.OL", "KOG.OL", "AKRBP.OL", "NHY.OL", "ORK.OL", "STB.OL", "YAR.OL",
    "WAWI.OL", "FRO.OL", "MOWI.OL", "TEL.OL", "SALM.OL", "SUBC.OL", "TOM.OL", "AUTO.OL",
    "GJF.OL", "BWLPG.OL", "BWE.OL", "SCHA.OL", "LSG.OL", "NAS.OL", "PGS.OL", "ODL.OL",
    "BAKKA.OL", "KIT.OL", "BONHR.OL", "VEI.OL", "ELMRA.OL", "ULTI.OL", "XXL.OL", "NOD.OL",
    "ATEA.OL", "SPBK1.OL", "AFG.OL", "ENTRA.OL", "SCATC.OL", "MPCC.OL", "SB1NO.OL",
    "PARB.OL", "HAFNI.OL", "GOGL.OL", "ABG.OL", "PROT.OL", "VOW.OL", "RECSI.OL",
    "BORR.OL", "NEL.OL", "SOFF.OL", "HEX.OL", "KAHOT.OL"
]


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA100"] = df["Close"].rolling(100).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["RSI14"] = calculate_rsi(df["Close"], 14)
    df["AVG_VOL20"] = df["Volume"].rolling(20).mean()
    df["ATR14"] = calculate_atr(df, 14)
    df["ATR_PCT"] = (df["ATR14"] / df["Close"]) * 100
    df["HIGH20"] = df["High"].rolling(20).max()
    df["HIGH50"] = df["High"].rolling(50).max()
    df["LOW20"] = df["Low"].rolling(20).min()
    df["RET_20D"] = df["Close"].pct_change(20) * 100
    return df


def fetch_data(symbol: str) -> pd.DataFrame | None:
    try:
        df = yf.download(symbol, period=LOOKBACK_PERIOD, interval="1d", progress=False, auto_adjust=False)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return None

    return add_indicators(df.dropna(how="all"))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def score_setup(latest: pd.Series) -> dict:
    score = 0.0

    close_ = float(latest["Close"])
    ma20 = float(latest["MA20"])
    ma50 = float(latest["MA50"])
    ma100 = float(latest["MA100"])
    ma200 = float(latest["MA200"])
    rsi = float(latest["RSI14"])
    atr_pct = float(latest["ATR_PCT"])
    avg_vol20 = float(latest["AVG_VOL20"]) if not pd.isna(latest["AVG_VOL20"]) else 0.0
    volume = float(latest["Volume"])
    volume_ratio = (volume / avg_vol20) if avg_vol20 > 0 else 1.0

    if close_ > ma200:
        score += 1.5
    if ma20 > ma50:
        score += 1.5
    if ma50 > ma100:
        score += 1.0
    if ma100 > ma200:
        score += 1.0
    if close_ > ma20:
        score += 0.5

    if 52 <= rsi <= 67:
        score += 2.0
    elif 45 <= rsi < 52 or 67 < rsi <= 72:
        score += 1.0
    elif rsi > 80 or rsi < 35:
        score -= 1.0

    breakout20 = close_ >= float(latest["HIGH20"]) * 0.985
    breakout50 = close_ >= float(latest["HIGH50"]) * 0.985
    if breakout20:
        score += 1.0
    if breakout50:
        score += 1.0

    if volume_ratio >= 1.5:
        score += 1.5
    elif volume_ratio >= 1.2:
        score += 1.0

    if 1.5 <= atr_pct <= 6.5:
        score += 1.0
    elif atr_pct > 9:
        score -= 0.5

    distance_from_ma20 = ((close_ / ma20) - 1) * 100 if ma20 else 0.0
    if distance_from_ma20 > 8:
        score -= 0.5

    return {
        "score": round(clamp(score, 0.0, 10.0), 2),
        "volume_ratio": round(volume_ratio, 2),
        "atr_pct": round(atr_pct, 2),
        "breakout": bool(breakout20 or breakout50),
    }


def derive_levels(latest: pd.Series) -> dict:
    close_ = float(latest["Close"])
    atr = float(latest["ATR14"])
    ma20 = float(latest["MA20"])
    low20 = float(latest["LOW20"])

    entry = close_
    stop_candidate = min(close_ - (1.6 * atr), ma20 * 0.985, low20 * 0.995)
    stop_loss = min(stop_candidate, close_ * 0.97)
    stop_loss = max(stop_loss, close_ * 0.90)

    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        stop_loss = close_ * 0.96
        risk_per_share = entry - stop_loss

    target = entry + (risk_per_share * 2.0)
    rr = (target - entry) / risk_per_share if risk_per_share > 0 else 0.0

    return {
        "entry": round(entry, 2),
        "stop": round(stop_loss, 2),
        "target": round(target, 2),
        "rr": round(rr, 2),
        "risk_per_share": round(risk_per_share, 2),
    }


def determine_signal(score: float, rr: float, latest: pd.Series) -> str:
    bullish = latest["MA20"] > latest["MA50"] > latest["MA100"]
    above_ma200 = latest["Close"] > latest["MA200"]
    overbought = latest["RSI14"] >= 75
    broken = latest["Close"] < latest["MA50"]

    if bullish and above_ma200 and score >= MIN_SCORE_BUY and rr >= MIN_RR and not overbought:
        return "BUY"
    if broken:
        return "HOLD"
    if score >= MIN_SCORE_WATCH:
        return "WATCH"
    return "HOLD"


def calculate_position(entry: float, stop: float) -> tuple[int, float]:
    max_risk_nok = START_CAPITAL_NOK * RISK_PER_TRADE
    max_position_value = START_CAPITAL_NOK * MAX_POSITION_PCT
    risk_per_share = entry - stop

    if risk_per_share <= 0:
        return 0, 0.0

    shares_by_risk = int(max_risk_nok // risk_per_share)
    shares_by_capital = int(max_position_value // entry)
    shares = max(0, min(shares_by_risk, shares_by_capital))
    value = round(shares * entry, 2)
    return shares, value


def build_comment(symbol: str, signal: str, latest: pd.Series, breakout: bool, volume_ratio: float) -> str:
    bullish = latest["MA20"] > latest["MA50"] > latest["MA100"]
    trend = "sterk trend" if bullish else "blandet trend"
    breakout_text = "nær breakout" if breakout else "under breakout-nivå"
    volume_text = "sterkt volum" if volume_ratio >= 1.2 else "normalt volum"

    if signal == "BUY":
        return f"{symbol}: {trend}, RSI {latest['RSI14']:.1f}, {volume_text}, {breakout_text}."
    if signal == "WATCH":
        return f"{symbol}: interessant oppsett, men trenger bekreftelse. RSI {latest['RSI14']:.1f}, {volume_text}."
    return f"{symbol}: ikke et rent oppsett nå. {trend}, RSI {latest['RSI14']:.1f}."


def analyze_symbol(symbol: str) -> dict | None:
    df = fetch_data(symbol)
    if df is None or len(df) < 220:
        return None

    latest = df.iloc[-1]
    needed = ["MA20", "MA50", "MA100", "MA200", "RSI14", "AVG_VOL20", "ATR14", "RET_20D"]
    if pd.isna(latest[needed]).any():
        return None

    score_info = score_setup(latest)
    levels = derive_levels(latest)
    signal = determine_signal(score_info["score"], levels["rr"], latest)
    shares, position_value = calculate_position(levels["entry"], levels["stop"])
    comment = build_comment(symbol, signal, latest, score_info["breakout"], score_info["volume_ratio"])

    return {
        "symbol": symbol,
        "signal": signal,
        "score": score_info["score"],
        "entry": levels["entry"],
        "stop": levels["stop"],
        "target": levels["target"],
        "rr": levels["rr"],
        "rsi": round(float(latest["RSI14"]), 2),
        "volumeRatio": score_info["volume_ratio"],
        "change20d": round(float(latest["RET_20D"]), 2),
        "positionShares": shares,
        "positionValue": position_value,
        "comment": comment,
        "scanDate": datetime.utcnow().isoformat() + "Z",
    }


def main():
    results = []

    for symbol in OSLO_TICKERS:
        print(f"Scanning {symbol}...")
        result = analyze_symbol(symbol)
        if result is not None:
            results.append(result)

    results.sort(
        key=lambda x: (
            {"BUY": 0, "WATCH": 1, "HOLD": 2}.get(x["signal"], 3),
            -x["score"],
            -x["rr"],
        )
    )

    top_results = results[:TOP_N]

    Path("signals.json").write_text(
        json.dumps(top_results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    summary = {
        "updatedAt": datetime.utcnow().isoformat() + "Z",
        "count": len(top_results),
        "buyCount": len([x for x in top_results if x["signal"] == "BUY"]),
        "watchCount": len([x for x in top_results if x["signal"] == "WATCH"]),
        "holdCount": len([x for x in top_results if x["signal"] == "HOLD"]),
    }

    Path("summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Done. signals.json updated.")


if __name__ == "__main__":
    main()
