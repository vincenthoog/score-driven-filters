import pandas as pd
import numpy as np
from io import StringIO
import re


def get_data(check, freq):  # freq in {'daily','monthly'}
    if freq == 'daily':
        path    = '49_Industry_Portfolios_Daily.csv'
        rf_path = 'F_F_Research_Data_Factors_Daily.csv'
        VW_MARK = 'Average Value Weighted Returns -- Daily'
        EW_MARK = 'Average Equal Weighted Returns -- Daily'
        start_dt = pd.Timestamp('1969-07-01')
    elif freq == 'monthly':
        path    = '49_Industry_Portfolios.csv'
        rf_path = 'F_F_Research_Data_Factors.csv'
        VW_MARK = 'Average Value Weighted Returns -- Monthly'
        EW_MARK = 'Average Equal Weighted Returns -- Monthly'
        start_dt = pd.Timestamp('1969-07-01')
    else:
        raise ValueError("freq must be 'daily' or 'monthly'")

    def first_cell(s: str) -> str:
        return s.strip().split(",")[0] if "," in s else s.strip()

    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()


    markers = []
    for i, ln in enumerate(lines):
        if VW_MARK in ln:
            markers.append(('VW', i))
        if EW_MARK in ln:
            markers.append(('EW', i))
    markers.sort(key=lambda x: x[1])
    if len(markers) < 2:
        raise RuntimeError("Could not find both VW and EW markers.")

    def find_block(tag):
        ks = [k for k,(t,_) in enumerate(markers) if t == tag]
        if not ks:
            raise RuntimeError(f"Marker {tag} not found.")
        k = ks[0]
        start_idx = markers[k][1]
        end_idx   = markers[k+1][1] if k+1 < len(markers) else len(lines)
        i_head = None
        for j in range(start_idx, end_idx):
            if first_cell(lines[j]).strip().lower() == "date":
                i_head = j
                break
        if i_head is None:
            raise RuntimeError(f"Couldn't find 'Date' header after {tag} marker.")
        return i_head, end_idx

    def parse_panel_by_marker(tag):
        i_head, end_idx = find_block(tag)
        block = "\n".join(lines[i_head:end_idx])
        df = pd.read_csv(StringIO(block), engine="python", dtype=str)
        df.columns = [c.strip() for c in df.columns]
        if "Date" not in df.columns:
            raise KeyError(f"No 'Date' column for {tag}: {df.columns.tolist()}")


        tok = df["Date"].astype(str).str.extract(r'(\d{6,8})')[0]
        df = df.assign(_tok=tok)
        mask6 = df["_tok"].str.len() == 6
        mask8 = df["_tok"].str.len() == 8

        if mask8.any():
            df = df[mask8].copy()
            df["_date"] = pd.to_datetime(df["_tok"], format="%Y%m%d", errors="coerce")
        elif mask6.any():
            df = df[mask6].copy()
            d6 = pd.to_datetime(df["_tok"], format="%Y%m", errors="coerce")
            df["_date"] = d6.dt.to_period("M").dt.to_timestamp("M")
        else:
            raise RuntimeError(f"{tag} panel: no recognizable YYYYMM or YYYYMMDD dates.")

        df = df.dropna(subset=["_date"]).set_index("_date").sort_index()
        df = df.drop(columns=["_tok"])
        for c in df.columns:
            if c != "Date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.drop(columns=["Date"]).dropna(how="all")
        return df

    vw = parse_panel_by_marker('VW')
    ew = parse_panel_by_marker('EW')

    vw_sub = vw.loc[vw.index >= start_dt].replace(-99.99, np.nan) / 100.0
    ew_sub = ew.loc[ew.index >= start_dt].replace(-99.99, np.nan) / 100.0

    if check:
        print(f"{freq.upper()} VW subset:", vw_sub.index.min(), "->", vw_sub.index.max(), vw_sub.shape)
        print(f"{freq.upper()} EW subset:", ew_sub.index.min(), "->", ew_sub.index.max(), ew_sub.shape)

    def parse_ff_rf(rf_path, expected=freq, start="1969-07-01"):
        with open(rf_path, "r", encoding="utf-8-sig") as f:
            rlines = f.read().splitlines()
        hdr = None
        for i, ln in enumerate(rlines):
            if re.search(r"\bDate\b", ln, flags=re.I) and re.search(r"\bRF\b", ln, flags=re.I):
                hdr = i
                break
        if hdr is None:
            raise RuntimeError("Could not find header with 'Date' and 'RF' in RF file.")
        rf = pd.read_csv(StringIO("\n".join(rlines[hdr:])), dtype=str)
        rf.columns = [c.strip() for c in rf.columns]

        tok = rf["Date"].astype(str).str.extract(r'(\d{6,8})')[0]
        rf = rf.assign(_tok=tok)
        mask6 = rf["_tok"].str.len() == 6
        mask8 = rf["_tok"].str.len() == 8

        if expected == 'daily' and mask8.any():
            rf = rf[mask8].copy()
            rf["_date"] = pd.to_datetime(rf["_tok"], format="%Y%m%d", errors="coerce")
        elif expected == 'monthly' and mask6.any():
            rf = rf[mask6].copy()
            d6 = pd.to_datetime(rf["_tok"], format="%Y%m", errors="coerce")
            rf["_date"] = d6.dt.to_period("M").dt.to_timestamp("M")
        else:
            if mask8.any():
                rf = rf[mask8].copy()
                rf["_date"] = pd.to_datetime(rf["_tok"], format="%Y%m%d", errors="coerce")
            elif mask6.any():
                rf = rf[mask6].copy()
                d6 = pd.to_datetime(rf["_tok"], format="%Y%m", errors="coerce")
                rf["_date"] = d6.dt.to_period("M").dt.to_timestamp("M")
            else:
                raise RuntimeError("RF file has no recognizable YYYYMM/YYYYMMDD dates.")

        rf["RF"] = pd.to_numeric(rf["RF"], errors="coerce")
        rf = rf.dropna(subset=["_date","RF"]).set_index("_date").sort_index()
        rf_dec = (rf[["RF"]] / 100.0).rename(columns={"RF":"rf"})
        rf_dec = rf_dec.loc[pd.to_datetime(start):]
        return rf_dec

    rf_dec = parse_ff_rf(rf_path, expected=freq, start="1969-07-01")

    end = vw_sub.index.max()
    rf_cut = rf_dec.loc[:end]
    vw_xs = vw_sub.join(rf_cut, how="inner")
    vw_xs = vw_xs.drop(columns="rf").sub(vw_xs["rf"], axis=0)

    ew_xs = ew_sub.join(rf_cut, how="inner")
    ew_xs = ew_xs.drop(columns="rf").sub(ew_xs["rf"], axis=0)

    if check:
        print("RF range:", rf_cut.index.min(), "->", rf_cut.index.max(), rf_cut.shape)
        print(vw_xs.head(3))
    return vw_xs, ew_xs

# for later: make sure all correlation ones workk
def build_W_from_corr(
    X,
    k = 6,
    min_corr = 0.10, 
    standardize = True,
    corr_method = "pearson",    
    winsor_pct = 0.02,         
    mcd_support_fraction = None,  
    mcd_random_state = 0,
    eps = 1e-12,
):
    
    cols = list(X.columns)
    R = len(cols)

    if standardize:
        Z = (X - X.mean()) / X.std(ddof=1)
    else:
        Z = X.copy()

    method = corr_method.lower()

    C_df = Z.corr(method=method)
    C = C_df.to_numpy()

    np.fill_diagonal(C, 0.0)

    if min_corr > 0:
        C = np.where(C >= float(min_corr), C, 0.0)
    else:
        C = np.where(C >= 0.0, C, 0.0)

    k_eff = max(1, min(int(k), R - 1))
    idx = np.argpartition(-C, k_eff - 1, axis=1)[:, :k_eff]
    rows = np.repeat(np.arange(R), k_eff)
    Wd = np.zeros_like(C)
    Wd[rows, idx.ravel()] = C[rows, idx.ravel()]
    
    W = Wd

    rs = W.sum(axis=1, keepdims=True)
    W = np.divide(W, rs + eps)         
    np.fill_diagonal(W, 0.0)

    return pd.DataFrame(W, index=cols, columns=cols)

