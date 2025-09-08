# sheets_bootstrap_min.py
import os, json, time
import gspread

def _client():
    sa = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    return gspread.service_account_from_dict(sa)

def _open():
    return _client().open_by_key(os.environ["GSHEET_ID"])

def _retry(fn, *args, **kwargs):
    delay = 1.0
    for _ in range(5):
        try:
            return fn(*args, **kwargs)
        except Exception:
            time.sleep(delay)
            delay = min(delay*2, 10)
    # tente une dernière fois
    return fn(*args, **kwargs)

def setup():
    sh = _open()

    # 1) Onglet trades : ajoute la colonne calculée K = pnl_num (nombre)
    ws_trades = _retry(sh.worksheet, "trades")
    # En-tête
    _retry(ws_trades.update, "K1", [["pnl_num"]], value_input_option="USER_ENTERED")
    # ARRAYFORMULA pour convertir "1,23" -> 1.23 puis nombre
    _retry(
        ws_trades.update,
        "K2",
        [[
            '=ARRAYFORMULA(IF(ROW(I:I)=1,"pnl_num",IF(I:I="","",VALUE(SUBSTITUTE(I:I, ",",".")))))'
        ]],
        value_input_option="USER_ENTERED"
    )

    # 2) Onglet bilan_journalier : PnL net par date (SELL uniquement)
    try:
        ws_daily = _retry(sh.worksheet, "bilan_journalier")
    except gspread.WorksheetNotFound:
        ws_daily = _retry(sh.add_worksheet, title="bilan_journalier", rows=2000, cols=5)

    # Formule (locale FR) : somme du PnL par date, en ne gardant que les SELL
    formula = (
        '=QUERY({trades!B:B,trades!K:K,trades!C:C},'
        '"select Col1, sum(Col2) '
        ' where Col3=''SELL'' and Col1 is not null '
        ' group by Col1 order by Col1 '
        ' label Col1 ''date'', sum(Col2) ''PnL_net''",1)'
    )
    _retry(ws_daily.update, "A1", [[formula]], value_input_option="USER_ENTERED")
