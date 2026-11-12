"""Effetti descrittivi e baseline stilometriche con gruppi di fonte disgiunti.

La fonte ``pair_id`` è l'unità di ricampionamento e partizionamento. I documenti
umani e tutti gli output dei generatori derivati dalla stessa fonte devono
condividere questo ID. Il segno degli effetti è sempre macchina meno umano.

"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _validate(df: pd.DataFrame, features: Sequence[str]) -> None:
    required = {"id", "pair_id", "domain", "generator", "label"}
    missing = required.union(features).difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df["id"].duplicated().any():
        raise ValueError("id must identify each unique document")
    if df["pair_id"].isna().any() or df["pair_id"].astype(str).eq("").any():
        raise ValueError("Every document requires a source pair_id")
    if "group_id" in df and df.group_id.isna().any():
        raise ValueError("group_id must be present for every document when supplied")
    if "group_id" in df and (df.groupby("pair_id").group_id.nunique() > 1).any():
        raise ValueError("All texts from a source pair_id must share one group_id")
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Both labels 0 (human) and 1 (machine) are required")


def _group_column(df: pd.DataFrame) -> str:
    """Le componenti più ampie collegate da duplicati hanno precedenza sulle coppie di fonte."""
    return "group_id" if "group_id" in df else "pair_id"


def _bh(p: np.ndarray) -> np.ndarray:
    """Correzione Benjamini-Hochberg sui test con valori finiti entro un singolo ambito."""
    q = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
        q[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return q


def _g_from_moments(n0, s0, ss0, n1, s1, ss1):
    """g di Hedges vettorizzato; non definito se un n<2 o la deviazione standard aggregata è nulla."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mean0, mean1 = s0 / n0, s1 / n1
        var0 = np.maximum((ss0 - s0**2 / n0) / (n0 - 1), 0)
        var1 = np.maximum((ss1 - s1**2 / n1) / (n1 - 1), 0)
        dof = n0 + n1 - 2
        pooled = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / dof)
        g = (1 - 3 / (4 * dof - 1)) * (mean1 - mean0) / pooled
    valid = (n0 >= 2) & (n1 >= 2) & (pooled > 0)
    return np.where(valid, g, np.nan)


def _paired_generator_subset(df: pd.DataFrame, generator: str) -> pd.DataFrame:
    """Confronta un generatore con gli umani dei soli gruppi di fonte per cui è disponibile."""
    machines = df.loc[(df.label == 1) & (df.generator == generator)]
    groups = set(machines.pair_id)
    return df.loc[((df.label == 0) & df.pair_id.isin(groups)) | df.index.isin(machines.index)]


def _effect_scope(
    df: pd.DataFrame,
    features: list[str],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    x = df[features].to_numpy(dtype=float)
    x[~np.isfinite(x)] = np.nan
    y = df.label.to_numpy()
    moments = []
    for label in (0, 1):
        z = x[y == label]
        finite = np.isfinite(z)
        counts = finite.sum(axis=0).astype(float)
        sums = np.nansum(z, axis=0)
        sums_sq = np.nansum(z**2, axis=0)
        moments.append((counts, sums, sums_sq))
    g = _g_from_moments(*moments[0], *moments[1])
    means, sds = [], []
    for n, sums, sums_sq in moments:
        with np.errstate(divide="ignore", invalid="ignore"):
            means.append(sums / n)
            sds.append(np.sqrt(np.maximum((sums_sq - sums**2 / n) / (n - 1), 0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p = np.asarray(stats.ttest_ind(x[y == 1], x[y == 0], equal_var=False, nan_policy="omit").pvalue)
    both_constant = (sds[0] == 0) & (sds[1] == 0)
    p[both_constant] = np.where(means[0][both_constant] == means[1][both_constant], 1.0, 0.0)
    pair_means = []
    for label in (0, 1):
        values = df.loc[df.label == label, ["pair_id", *features]].copy()
        values[features] = values[features].replace([np.inf, -np.inf], np.nan)
        pair_means.append(values.groupby("pair_id")[features].mean())
    delta = pair_means[1] - pair_means[0]
    n_paired = delta.notna().sum().to_numpy()
    test_delta = delta
    if "group_id" in df:
        pair_groups = df.drop_duplicates("pair_id").set_index("pair_id").group_id
        test_delta = delta.groupby(pair_groups.reindex(delta.index)).mean()
    n_test_groups = test_delta.notna().sum().to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        paired_p = np.asarray(stats.ttest_1samp(test_delta.to_numpy(), 0, axis=0, nan_policy="omit").pvalue)
    constant_delta = (test_delta.std(ddof=1).to_numpy() == 0) & (n_test_groups >= 2)
    paired_p[constant_delta] = np.where(test_delta.mean().to_numpy()[constant_delta] == 0, 1.0, 0.0)
    paired = n_test_groups >= 2
    p[paired] = paired_p[paired]
    insufficient_paired = (n_paired > 0) & ~paired
    p[insufficient_paired] = np.nan
    low = np.full(len(features), np.nan)
    high = low.copy()
    valid_boot = np.zeros(len(features), dtype=int)
    groups, group_codes = np.unique(df[_group_column(df)].astype(str), return_inverse=True)
    if n_bootstrap > 0 and len(groups) >= 2:
        weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=n_bootstrap)
        boot_moments = []
        for label in (0, 1):
            mask = y == label
            values = np.nan_to_num(x[mask], nan=0.0)
            arrays = []
            for val in (np.isfinite(x[mask]).astype(float), values, values**2):
                grouped = np.zeros((len(groups), len(features)), dtype=float)
                np.add.at(grouped, group_codes[mask], val)
                arrays.append(weights @ grouped)
            boot_moments.append(arrays)
        draws = _g_from_moments(*boot_moments[0], *boot_moments[1])
        valid_boot = np.isfinite(draws).sum(axis=0)
        for j in range(len(features)):
            finite = draws[np.isfinite(draws[:, j]), j]
            if len(finite) >= max(20, n_bootstrap // 2):
                low[j], high[j] = np.quantile(finite, [0.025, 0.975])
    return pd.DataFrame({
        "feature": features,
        "n_human": moments[0][0].astype(int),
        "n_machine": moments[1][0].astype(int),
        "n_source_groups": len(groups),
        "n_paired": n_paired,
        "n_test_groups": n_test_groups,
        "human_mean": means[0],
        "machine_mean": means[1],
        "human_sd": sds[0],
        "machine_sd": sds[1],
        "hedges_g": g,
        "ci_low": low,
        "ci_high": high,
        "bootstrap_valid": valid_boot,
        "test_type": np.where(paired, np.where(n_test_groups < n_paired, "paired_cluster_t", "paired_t"),
                              np.where(insufficient_paired, "insufficient_independent_pairs", "welch_unpaired")),
        "p_value": p,
        "bh_q": _bh(p),
    })


def effect_tables(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    seed: int = 42,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """Effetti aggregati e distinti per dominio, generatore e combinazione dei due.

    Gli intervalli sono percentili al 95% ottenuti da bootstrap per gruppi di fonte.
    I p-value usano test t appaiati sulle differenze macchina-umano per fonte quando
    esistono almeno due gruppi indipendenti completi; se sono fornite componenti
    ``group_id`` collegate da duplicati, le differenze appaiate sono prima mediate
    al loro interno. Altrimenti viene usato un test di Welch non appaiato,
    esplicitamente segnalato. La correzione BH è applicata alle feature separatamente
    in ogni ambito mostrato; non si dichiara una correzione fra le diverse tabelle.
    """
    features = list(feature_columns)
    _validate(df, features)
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    tables = []
    scopes = [("pooled", "ALL", "ALL", df)]
    scopes.extend(("domain", str(d), "ALL", sub) for d, sub in df.groupby("domain", sort=True))
    generators = sorted(df.loc[df.label == 1, "generator"].unique())
    for generator in generators:
        sub = _paired_generator_subset(df, generator)
        scopes.append(("generator", "ALL", str(generator), sub))
        scopes.extend(("domain_generator", str(d), str(generator), part)
                      for d, part in sub.groupby("domain", sort=True))
    for scope, domain, generator, sub in scopes:
        if sub.label.nunique() < 2:
            continue
        table = _effect_scope(sub, features, rng, n_bootstrap)
        table.insert(0, "generator", generator)
        table.insert(0, "domain", domain)
        table.insert(0, "scope", scope)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _folds(df: pd.DataFrame, seed: int, n_splits: int):
    group_col = _group_column(df)
    n_groups = df[group_col].nunique()
    if n_groups < n_splits:
        raise ValueError(f"Need at least {n_splits} source groups; found {n_groups}")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(splitter.split(df, df.label, df[group_col].astype(str))):
        yield fold, df.iloc[train].index.to_numpy(), df.iloc[test].index.to_numpy()


def _evaluation_splits(df: pd.DataFrame, seed: int, n_splits: int):
    group_col = _group_column(df)
    for domain, sub in df.groupby("domain", sort=True):
        for fold, train, test in _folds(sub, seed, n_splits):
            yield "within_domain", str(domain), "ALL", fold, train, test
    for domain in sorted(df.domain.unique()):
        test_mask = df.domain == domain
        test_groups = set(df.loc[test_mask, group_col])
        train_mask = ~test_mask & ~df[group_col].isin(test_groups)
        yield "leave_domain_out", str(domain), "ALL", 0, df.index[train_mask].to_numpy(), df.index[test_mask].to_numpy()
    for generator in sorted(df.loc[df.label == 1, "generator"].unique()):
        target = _paired_generator_subset(df, generator)
        for fold, _, test in _folds(target, seed, n_splits):
            test_groups = set(df.loc[test, group_col])
            train_mask = ~df[group_col].isin(test_groups) & ((df.label == 0) | (df.generator != generator))
            yield "leave_generator_out", "ALL", str(generator), fold, df.index[train_mask].to_numpy(), test


def evaluate(
    df: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    seed: int = 42,
    n_splits: int = 5,
    n_bootstrap: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Addestra modelli logistici L2 fissi e restituisce metriche e predizioni fuori fold.

    Ogni fit include imputazione con mediana, StandardScaler e LogisticRegression
    (C=1, class_weight='balanced', soglia=0.5, max_iter=2000). Iperparametri e insiemi
    di feature sono fissi prima del test. Le metriche aggregano predizioni fuori
    fold per dominio o generatore escluso; i fold non sono campioni indipendenti.
    Gli intervalli usano bootstrap per gruppi di fonte con predizioni fuori fold
    fisse: non misurano tutta l'incertezza di training o selezione del dataset.
    """
    features = list(dict.fromkeys(f for values in feature_sets.values() for f in values))
    _validate(df, features)
    if not features:
        raise ValueError("At least one feature is required")
    data = df.reset_index(drop=True).copy()
    group_col = _group_column(data)
    predictions = []
    for protocol, domain, generator, fold, train, test in _evaluation_splits(data, seed, n_splits):
        train_groups, test_groups = set(data.loc[train, group_col]), set(data.loc[test, group_col])
        if train_groups.intersection(test_groups):
            raise AssertionError("Source group leakage")
        if data.loc[train, "label"].nunique() != 2 or data.loc[test, "label"].nunique() != 2:
            raise ValueError(f"Both classes required in {protocol}/{domain}/{generator}/fold{fold}")
        for name, columns in feature_sets.items():
            columns = list(columns)
            if not columns:
                continue
            model = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
            )
            x_train = data.loc[train, columns].replace([np.inf, -np.inf], np.nan)
            x_test = data.loc[test, columns].replace([np.inf, -np.inf], np.nan)
            model.fit(x_train, data.loc[train, "label"])
            proba = model.predict_proba(x_test)[:, 1]
            metadata = list(dict.fromkeys(["id", "pair_id", group_col, "domain", "generator", "label"]))
            out = data.loc[test, metadata].copy()
            out.rename(columns={"domain": "text_domain", "generator": "text_generator"}, inplace=True)
            out["protocol"], out["domain"], out["generator"] = protocol, domain, generator
            out["feature_set"], out["fold"] = name, fold
            out["p_machine"], out["prediction"] = proba, (proba >= 0.5).astype(int)
            out["n_train"] = len(train)
            out["n_features"] = len(columns)
            out["train_group_overlap"] = len(train_groups.intersection(test_groups))
            predictions.append(out)
    preds = pd.concat(predictions, ignore_index=True)
    rng = np.random.default_rng(seed + 1307)
    metrics = []
    keys = ["protocol", "domain", "generator", "feature_set"]
    for values, part in preds.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        if part.id.duplicated().any():
            raise AssertionError(f"A document has duplicate OOF predictions in {values}")
        row.update(_score(part))
        row.update({"n_test": len(part), "n_human": int((part.label == 0).sum()),
                    "n_machine": int((part.label == 1).sum()), "n_folds": part.fold.nunique(),
                    "n_train_min": int(part.n_train.min()), "n_train_max": int(part.n_train.max()),
                    "n_features": int(part.n_features.iloc[0])})
        boot = _metric_bootstrap(part, rng, n_bootstrap)
        row.update(boot)
        metrics.append(row)
    return pd.DataFrame(metrics), preds


def _score(df: pd.DataFrame) -> dict:
    return {
        "roc_auc": roc_auc_score(df.label, df.p_machine),
        "balanced_accuracy": balanced_accuracy_score(df.label, df.prediction),
        "macro_f1": f1_score(df.label, df.prediction, average="macro", zero_division=0),
    }


def _metric_bootstrap(df: pd.DataFrame, rng, n_bootstrap: int) -> dict:
    group_col = _group_column(df)
    groups = df[group_col].astype(str).unique()
    index_groups = {g: np.flatnonzero(df[group_col].astype(str).to_numpy() == g) for g in groups}
    scores = []
    for _ in range(n_bootstrap):
        chosen = rng.choice(groups, len(groups), replace=True)
        idx = np.concatenate([index_groups[g] for g in chosen])
        sample = df.iloc[idx]
        if sample.label.nunique() == 2:
            scores.append(_score(sample))
    result = {}
    for key in ("roc_auc", "balanced_accuracy", "macro_f1"):
        values = [s[key] for s in scores]
        result[key + "_ci_low"], result[key + "_ci_high"] = np.quantile(values, [0.025, 0.975]) if values else (np.nan, np.nan)
    return result


def shortlist_features(
    df: pd.DataFrame,
    effects: pd.DataFrame,
    feature_columns: Sequence[str],
    min_abs_g: float = 0.3,
    min_consistency: float = 5 / 7,
    max_features: int = 12,
    max_correlation: float = 0.85,
) -> tuple[list[str], pd.DataFrame]:
    """Shortlist esplorativa da tutte le osservazioni.
    """
    rows = []
    for feature in feature_columns:
        parts = effects[effects.feature == feature]
        pooled = parts.loc[parts.scope == "pooled", "hedges_g"]
        sign = np.sign(pooled.iloc[0]) if len(pooled) else 0
        domain_g = parts.loc[parts.scope == "domain", "hedges_g"].to_numpy()
        generator_g = parts.loc[parts.scope == "generator", "hedges_g"].to_numpy()
        domains_pass = (np.isfinite(domain_g) & (domain_g * sign >= min_abs_g))
        generators_pass = (np.isfinite(generator_g) & (generator_g * sign > 0))
        fraction = generators_pass.mean() if len(generator_g) else 0.0
        eligible = bool(len(domain_g) and domains_pass.all() and fraction >= min_consistency)
        score = float(np.nanmedian(np.abs(domain_g)) * fraction) if len(domain_g) else 0.0
        rows.append({"feature": feature, "pooled_g": pooled.iloc[0] if len(pooled) else np.nan,
                     "domains_passing": int(domains_pass.sum()), "n_domains": len(domain_g),
                     "generators_passing": int(generators_pass.sum()), "n_generators": len(generator_g),
                     "generator_fraction": fraction, "min_domain_abs_g": float(np.min(np.abs(domain_g))) if len(domain_g) else np.nan,
                     "ranking_score": score, "eligible": eligible})
    robustness = pd.DataFrame(rows).sort_values(["eligible", "ranking_score"], ascending=False, kind="stable")
    selected = []
    reasons = {}
    corr = df[list(feature_columns)].corr(method="spearman").abs()
    for row in robustness.itertuples():
        if not row.eligible:
            reasons[row.feature] = "fails_domain_or_generator_rule"
        elif len(selected) >= max_features:
            reasons[row.feature] = "maximum_size"
        else:
            redundant = [name for name in selected if corr.loc[row.feature, name] >= max_correlation]
            if redundant:
                reasons[row.feature] = "correlated_with:" + redundant[0]
            else:
                selected.append(row.feature)
                reasons[row.feature] = "selected"
    robustness["selected"] = robustness.feature.isin(selected)
    robustness["selection_reason"] = robustness.feature.map(reasons)
    return selected, robustness.reset_index(drop=True)
