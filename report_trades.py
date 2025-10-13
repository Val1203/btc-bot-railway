#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génère un rapport propre (Excel) à partir d'un journal de trades,
avec corrections d'erreurs fréquentes et agrégations jour/semaine.

Entrées (via variables d’environnement, avec valeurs par défaut) :
- INPUT_PATH        : chemin vers le journal (CSV ou XLSX). Défaut: trades.csv
- SHEET_NAME        : si XLSX et onglet spécifique. Défaut: None (premier onglet)
- OUTPUT_XLSX       : chemin du fichier Excel de sortie. Défaut: weekly_report.xlsx
- TZ                : fuseau horaire IANA pour l’affichage (ex: Europe/Paris). Défaut: UTC
- DECIMAL_COMMA     : "1" si les nombres utilisent la virgule décimale. Défaut: auto
- FEES_PCT          : pourcentage de frais à appliquer par trade (ex: 0.1 pour 0,1%). Défaut: 0.0

Colonnes attendues (flexibles, auto-détectées) :
- datetime / date / time (timestamp ISO, ex: 2025-10-11T00:01:23Z)
- pair / symbol (ex: BTC/USDC)
- entry / buy_price / open_price
- exit / sell_price / close_price
- qty / amount (facultatif)
- fee / fees (facultatif, si absent on applique FEES_PCT)
- pnl_pct / roi (facultatif, sera recalculé)

Sorties :
- Excel avec 3 feuilles : Trades détaillés, Résumé quotidien, Résumé hebdo
"""

import os
import sys
import io
import math
import warnings
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np

warnings.simplefilter("ignore", category=UserWarning)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 2000)


# ---------- Utils ----------

def env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


def detect_decimal_comma(df: pd.DataFrame, candidate_cols: List[str]) -> bool:
    """Détecte si les nombres utilisent la virgule décimale."""
    for col in candidate_cols:
        if col in df.columns:
            sample = df[col].astype(str).dropna().head(50)
            # Si on voit beaucoup de virgules et peu de points, on suppose virgule décimale
            commas = sample.str.contains(",", regex=False).mean()
            dots = sample.str.contains(".", regex=False).mean()
            if commas > dots and commas > 0.2:
                return True
    return False


def to_numeric_safe(s: pd.Series, decimal_comma: bool) -> pd.Series:
    # Nettoie espaces, % et remplace virgule par point si besoin
    x = s.astype(str).str.replace(r"\s", "", regex=True).str.replace("%", "", regex=False)
    if decimal_comma:
        x = x.str.replace(",", ".", regex=False)
    # Vide, None, "nan" --> NaN
    x = x.replace({"": np.nan, "None": np.nan, "NaN": np.nan, "nan": np.nan})
    return pd.to_numeric(x, errors="coerce")


def parse_datetime(col: pd.Series) -> pd.Series:
    # Supporte ISO, avec/without Z ; si pas de tz, suppose UTC
    dt = pd.to_datetime(col, errors="coerce", utc=True, infer_datetime_format=True)
    return dt


def pick_first_existing(d: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in d.columns:
            return n
    return None


def calc_week_iso(dt_utc: pd.Series, tz: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
    # convertit en tz local pour affichage de date (jour). L’agrégation hebdo reste ISO (calendrier ISO = basé sur lundi)
    dt_local = dt_utc.dt.tz_convert(tz)
    day = dt_local.dt.strftime("%Y-%m-%d")
    # ISO week (YYYY-Www)
    week = dt_local.dt.isocalendar()
    week_iso = week["year"].astype(str) + "-W" + week["week"].astype(str).str.zfill(2)
    return dt_local, day, week_iso


def compute_pnl_pct(entry: pd.Series, exit_: pd.Series, fee_pct: float, fee_abs: Optional[pd.Series]) -> pd.Series:
    """
    PnL% net:
      brut = (exit - entry) / entry * 100
      frais:
        - si fee_abs présent et qty disponible on ne l’utilise pas ici (complexe), on applique fee_pct simplifié
        - sinon on applique fee_pct sur les deux jambes (aller + retour) -> approx: (2 * fee_pct)
    """
    brut = (exit_ - entry) / entry * 100.0
    if fee_abs is not None:
        # On ne connaît pas forcément la base, donc on garde fee_pct si fourni.
        pass
    net = brut - (2.0 * fee_pct)
    return net


def winloss_from_pnl(pnl_pct: pd.Series) -> pd.Series:
    lab = np.where(pnl_pct > 0.0, "WIN", np.where(pnl_pct < 0.0, "LOSS", "BE"))
    return pd.Series(lab, index=pnl_pct.index)


def summarize(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    g = df.groupby(key_cols, dropna=False)
    out = g.agg(
        trades=("pnl_pct", "count"),
        wins=("winloss", lambda s: (s == "WIN").sum()),
        losses=("winloss", lambda s: (s == "LOSS").sum()),
        be=("winloss", lambda s: (s == "BE").sum()),
        pnl_pct_total=("pnl_pct", "sum"),
    ).reset_index()
    out["winrate"] = np.where(out["trades"] > 0, out["wins"] / out["trades"] * 100.0, np.nan)
    # tri standard : date/sem puis pair
    order_cols = [c for c in key_cols if c in out.columns]
    out = out.sort_values(order_cols + ["trades"], ascending=[True] * len(order_cols) + [False])
    # arrondis lisibles
    for c in ["pnl_pct_total", "winrate"]:
        out[c] = out[c].round(4)
    return out


def load_input(path: str, sheet_name: Optional[str]) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    elif ext in [".csv", ".txt"]:
        # tentative intelligente d'encodage/sep
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=";")
    else:
        raise ValueError(f"Format non supporté: {ext}")


def main():
    # ---- Config depuis l'env ----
    INPUT_PATH = env("INPUT_PATH", "trades.csv")
    SHEET_NAME = env("SHEET_NAME", None)
    OUTPUT_XLSX = env("OUTPUT_XLSX", "weekly_report.xlsx")
    TZ = env("TZ", "UTC")
    FEES_PCT = float(env("FEES_PCT", "0.0"))
    DECIMAL_COMMA_ENV = env("DECIMAL_COMMA", None)  # "1" / "0" / None(auto)

    if not os.path.exists(INPUT_PATH):
        print(f"[ERREUR] Fichier introuvable: {INPUT_PATH}", file=sys.stderr)
        sys.exit(2)

    df_raw = load_input(INPUT_PATH, SHEET_NAME)

    # Normalise les noms de colonnes (minuscules, strip)
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    # Colonnes candidates
    col_dt = pick_first_existing(df_raw, ["datetime", "date", "time", "timestamp"])
    col_pair = pick_first_existing(df_raw, ["pair", "symbol", "market"])
    col_entry = pick_first_existing(df_raw, ["entry", "buy_price", "open_price", "entry_price"])
    col_exit = pick_first_existing(df_raw, ["exit", "sell_price", "close_price", "exit_price"])
    col_qty = pick_first_existing(df_raw, ["qty", "quantity", "amount", "size"])
    col_fee = pick_first_existing(df_raw, ["fee", "fees", "commission"])
    col_pnl_pct_in = pick_first_existing(df_raw, ["pnl_pct", "roi", "pnl%", "profit_pct"])

    missing = [("datetime", col_dt), ("pair/symbol", col_pair), ("entry", col_entry), ("exit", col_exit)]
    missing = [name for name, picked in missing if picked is None]
    if missing:
        print(f"[ERREUR] Colonnes manquantes: {', '.join(missing)}", file=sys.stderr)
        sys.exit(3)

    # Détection du séparateur décimal
    decimal_comma = False
    if DECIMAL_COMMA_ENV is None:
        decimal_comma = detect_decimal_comma(df_raw, [col_entry, col_exit, col_pnl_pct_in] if col_pnl_pct_in else [col_entry, col_exit])
    else:
        decimal_comma = DECIMAL_COMMA_ENV == "1"

    # Parsing des nombres
    df = df_raw.copy()
    df["_entry"] = to_numeric_safe(df[col_entry], decimal_comma)
    df["_exit"] = to_numeric_safe(df[col_exit], decimal_comma)
    if col_qty:
        df["_qty"] = to_numeric_safe(df[col_qty], decimal_comma)
    else:
        df["_qty"] = np.nan
    if col_fee:
        df["_fee_abs"] = to_numeric_safe(df[col_fee], decimal_comma)
    else:
        df["_fee_abs"] = np.nan

    # Parse datetime en UTC
    df["_dt_utc"] = parse_datetime(df[col_dt])
    # supprime lignes sans datetime ou prix
    before = len(df)
    df = df.dropna(subset=["_dt_utc", "_entry", "_exit"]).copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[INFO] Lignes invalides supprimées: {dropped}")

    # Convertit en fuseau voulu + colonnes auxiliaires
    dt_local, day_str, week_iso = calc_week_iso(df["_dt_utc"], TZ)
    df["datetime_local"] = dt_local.dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["day"] = day_str
    df["week_iso"] = week_iso

    # PnL %
    if col_pnl_pct_in and df_raw[col_pnl_pct_in].notna().any():
        pnl_in = to_numeric_safe(df_raw[col_pnl_pct_in], decimal_comma)
        df["pnl_pct"] = pnl_in
        # Recorrige si incohérent avec entry/exit (> 50% d'écart par ex.)
        pnl_recalc = compute_pnl_pct(df["_entry"], df["_exit"], FEES_PCT, df["_fee_abs"])
        mask_bad = (df["pnl_pct"].notna()) & (np.abs(df["pnl_pct"] - pnl_recalc) > 0.05)
        if mask_bad.any():
            df.loc[mask_bad, "pnl_pct"] = pnl_recalc[mask_bad]
    else:
        df["pnl_pct"] = compute_pnl_pct(df["_entry"], df["_exit"], FEES_PCT, df["_fee_abs"])

    df["winloss"] = winloss_from_pnl(df["pnl_pct"])

    # Harmonise le nom de la paire
    df["pair"] = df[col_pair]

    # Trie propre
    df = df.sort_values("_dt_utc").reset_index(drop=True)

    # Résumés
    daily = summarize(df, ["day", "pair"])
    weekly = summarize(df, ["week_iso", "pair"])
    # >>> ICI le bug est corrigé : weekly regroupe toutes les lignes partageant la même ISO semaine,
    #     pas seulement la même date.

    # Arrondis lisibles dans le détail
    for c in ["_entry", "_exit", "pnl_pct"]:
        df[c if c != "_entry" and c != "_exit" else c] = df[c].astype(float).round(6)
    df.rename(columns={"_entry": "entry", "_exit": "exit"}, inplace=True)

    # Colonnes finales ordonnées
    cols_out = [
        "datetime_local", "pair", "entry", "exit", "pnl_pct", "winloss", "day", "week_iso"
    ]
    if col_qty:
        cols_out.insert(3, "_qty")
        df.rename(columns={"_qty": "qty"}, inplace=True)
        cols_out[cols_out.index("_qty")] = "qty"
    if col_fee:
        cols_out.insert(cols_out.index("pnl_pct"), "_fee_abs")
        df.rename(columns={"_fee_abs": "fee_abs"}, inplace=True)
        cols_out[cols_out.index("_fee_abs")] = "fee_abs"

    detail = df[cols_out].copy()

    # Export Excel
    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        detail.to_excel(writer, index=False, sheet_name="Trades détaillés")
        daily.to_excel(writer, index=False, sheet_name="Résumé quotidien")
        weekly.to_excel(writer, index=False, sheet_name="Résumé hebdo")

        # Mise en forme simple
        wb = writer.book
        fmt_pct = wb.add_format({"num_format": "0.0000"})
        for sh, colnames in [("Trades détaillés", detail.columns),
                             ("Résumé quotidien", daily.columns),
                             ("Résumé hebdo", weekly.columns)]:
            ws = writer.sheets[sh]
            # largeur auto approximative
            for i, c in enumerate(colnames):
                width = max(10, min(28, int(detail[c].astype(str).str.len().mean() if c in detail.columns else 12) + 4))
                ws.set_column(i, i, width)
            # formats % sur colonnes pertinentes
            try:
                if "pnl_pct" in colnames:
                    j = list(colnames).index("pnl_pct")
                    ws.set_column(j, j, 12, fmt_pct)
                if "pnl_pct_total" in colnames:
                    j = list(colnames).index("pnl_pct_total")
                    ws.set_column(j, j, 14, fmt_pct)
                if "winrate" in colnames:
                    j = list(colnames).index("winrate")
                    ws.set_column(j, j, 10, fmt_pct)
            except Exception:
                pass

    print(f"[OK] Rapport généré : {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
