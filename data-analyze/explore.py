"""USPTO Patent Phrase-to-Phrase Matching — 数据分析与可视化工具集。

面向「微调」准备，提供对训练/测试数据的多维度探查与可视化函数。
所有函数均为纯函数：分析函数返回 pandas 对象 / dict；可视化函数创建
matplotlib.figure.Figure（可保存为图片，或在 notebook 中直接显示），
不在模块内联执行任何分析或绘制，便于在 notebook、Kaggle 脚本或 tests/ 中复用。

数据字段含义（Kaggle: US Patent Phrase to Phrase Matching）:
    id      : 样本唯一标识
    anchor  : 锚定短语（来自专利的词/短语）
    target  : 目标短语
    context : CPC 专利分类代码（如 G02、A47），作为领域上下文
    score   : anchor 与 target 在给定 context 下的语义相似度（0~1，离散取值）
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Optional

import pandas as pd

# 可视化依赖（无图形界面环境也能把图保存为文件）
import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 本地开发：data/ 目录；Kaggle 环境：/kaggle/input 下对应比赛目录
_LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_KAGGLE_CANDIDATES = [
    "/kaggle/input/us-patent-phrase-to-phrase-matching",
    "/kaggle/input/uspto-patent-phrase-to-phrase-matching",
]

EXPECTED_TRAIN_COLS = ["id", "anchor", "target", "context", "score"]
EXPECTED_TEST_COLS = ["id", "anchor", "target", "context"]


def resolve_data_path(filename: str) -> str:
    """按 本地 data/ -> Kaggle input 的顺序定位数据文件，返回绝对路径。"""
    local_path = os.path.join(_LOCAL_DATA_DIR, filename)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    for base in _KAGGLE_CANDIDATES:
        cand = os.path.join(base, filename)
        if os.path.exists(cand) and os.path.getsize(cand) > 0:
            return cand
    # 退化为本地路径（文件可能为空/不存在，由调用方处理）
    return local_path


def load_train(filename: str = "train.csv", fallback: str = "train_data_sample.csv") -> pd.DataFrame:
    """加载训练数据。

    优先使用 ``filename``（默认 train.csv）。若文件为空或不存在，
    则尝试 ``fallback``（样本文件）。若样本也为空/不存在，则读取 filename
    （让其正常抛出可读错误）。
    """
    path = resolve_data_path(filename)
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        alt = resolve_data_path(fallback)
        if os.path.exists(alt) and os.path.getsize(alt) > 0:
            path = alt
    df = pd.read_csv(path)
    if "score" not in df.columns:
        raise ValueError(f"训练数据缺少 'score' 列，疑似样本文件为空。请确认 {path} 内容。")
    return df


def load_test(filename: str = "test.csv") -> pd.DataFrame:
    """加载测试数据。"""
    return pd.read_csv(resolve_data_path(filename))


def _non_empty(df: pd.DataFrame, name: str = "数据") -> None:
    if df is None or df.empty:
        raise ValueError(f"{name}为空，无法分析。")


def basic_info(df: pd.DataFrame) -> pd.DataFrame:
    """返回每列的基础信息：非空数、缺失数、唯一值数、类型。"""
    _non_empty(df)
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "nulls": int(series.isna().sum()),
                "n_unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def id_integrity(df: pd.DataFrame) -> dict[str, int]:
    """检查 id 唯一性与重复行，返回统计 dict（用于检测泄漏/重复）。"""
    _non_empty(df)
    total = len(df)
    n_unique_id = df["id"].nunique(dropna=True)
    dup_rows = int(df.duplicated().sum())
    return {
        "total_rows": total,
        "unique_ids": int(n_unique_id),
        "duplicate_ids": int(total - n_unique_id),
        "duplicate_rows": dup_rows,
    }


def score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """score 取值分布（计数 + 占比），按占比降序。"""
    _non_empty(df)
    vc = df["score"].value_counts(dropna=False).sort_index()
    out = pd.DataFrame({"score": vc.index, "count": vc.values})
    out["proportion"] = (out["count"] / len(df)).round(4)
    return out


def context_distribution(df: pd.DataFrame, top_n: Optional[int] = 20) -> pd.DataFrame:
    """context（CPC 分类）取值分布，按样本数降序。"""
    _non_empty(df)
    vc = df["context"].value_counts(dropna=False)
    out = pd.DataFrame({"context": vc.index, "count": vc.values})
    out["proportion"] = (out["count"] / len(df)).round(4)
    return out.head(top_n) if top_n else out


def score_by_context(df: pd.DataFrame) -> pd.DataFrame:
    """每个 context 下的样本数、平均/中位/最小/最大 score。

    用于判断哪些专利领域更难（平均相似度低）或标注更集中。
    """
    _non_empty(df)
    g = df.groupby("context")["score"]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "mean_score": g.mean().round(4),
            "median_score": g.median(),
            "min_score": g.min(),
            "max_score": g.max(),
        }
    ).sort_values("n", ascending=False)
    return out


def phrase_stats(df: pd.DataFrame, columns: Iterable[str] = ("anchor", "target")) -> pd.DataFrame:
    """锚定/目标短语的统计：唯一短语数、Top 高频短语。"""
    _non_empty(df)
    rows = []
    for col in columns:
        vc = df[col].value_counts()
        rows.append(
            {
                "column": col,
                "n_unique": int(df[col].nunique()),
                "top_phrase": vc.index[0] if len(vc) else None,
                "top_count": int(vc.iloc[0]) if len(vc) else 0,
            }
        )
    return pd.DataFrame(rows)


def length_stats(df: pd.DataFrame, columns: Iterable[str] = ("anchor", "target")) -> pd.DataFrame:
    """各文本列的字数/词数分布（min/median/mean/max/p95）。

    微调输入截断（max_seq_len）的关键参考：据此设定 tokenizer 的最大长度。
    """
    _non_empty(df)
    rows = []
    for col in columns:
        char_len = df[col].astype(str).str.len()
        word_len = df[col].astype(str).str.split().map(len)
        rows.append(
            {
                "column": col,
                "char_min": int(char_len.min()),
                "char_median": float(char_len.median()),
                "char_mean": round(float(char_len.mean()), 2),
                "char_max": int(char_len.max()),
                "char_p95": float(char_len.quantile(0.95)),
                "word_max": int(word_len.max()),
                "word_p95": float(word_len.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def _tokenize(text: str) -> set[str]:
    return set(str(text).lower().split())


def _jaccard(anchor: str, target: str) -> float:
    a, t = _tokenize(anchor), _tokenize(target)
    union = len(a | t)
    if union == 0:
        return 0.0
    return len(a & t) / union


def token_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """计算 anchor 与 target 的词集合重叠度（Jaccard），并按 score 分组汇总。

    用于观察「文本重叠」与「相似度评分」的相关性，辅助构造特征或理解标签。
    """
    _non_empty(df)
    jaccard = df.apply(lambda r: _jaccard(r["anchor"], r["target"]), axis=1)
    tmp = pd.DataFrame({"score": df["score"].to_numpy(), "jaccard": jaccard.to_numpy()})
    g = tmp.groupby("score")["jaccard"]
    out = pd.DataFrame(
        {
            "mean": g.mean().round(4),
            "median": g.median().round(4),
            "count": g.size(),
        }
    )
    return out


def exact_match_rate(df: pd.DataFrame) -> dict[str, object]:
    """anchor 与 target 完全相同的样本占比，以及这些样本的平均 score。

    帮助理解标签是否（部分）由字面相等决定。
    """
    _non_empty(df)
    eq = (df["anchor"].astype(str) == df["target"].astype(str))
    return {
        "exact_match_count": int(eq.sum()),
        "exact_match_rate": round(float(eq.mean()), 4),
        "mean_score_when_equal": round(float(df.loc[eq, "score"].mean()), 4) if eq.any() else None,
        "mean_score_when_diff": round(float(df.loc[~eq, "score"].mean()), 4) if (~eq).any() else None,
    }


def summary_report(df: pd.DataFrame) -> dict[str, object]:
    """汇总运行上述所有分析，返回 {分析名: 结果} 的 dict。"""
    _non_empty(df)
    return {
        "basic_info": basic_info(df),
        "id_integrity": id_integrity(df),
        "score_distribution": score_distribution(df),
        "context_distribution": context_distribution(df),
        "score_by_context": score_by_context(df),
        "phrase_stats": phrase_stats(df),
        "length_stats": length_stats(df),
        "token_overlap": token_overlap(df),
        "exact_match_rate": exact_match_rate(df),
    }


# --------------------------------------------------------------------------- #
# 可视化（基于 seaborn）
# 约定：每个函数返回 matplotlib.figure.Figure；通过 save_path 保存为图片，
#      show=True 时在交互式环境中弹出显示（无界面环境下会被安全忽略）。
#      图内标签使用英文，避免 matplotlib 默认字体缺失中文导致乱码。
# --------------------------------------------------------------------------- #

def _finalize(fig: plt.Figure, save_path: Optional[str], show: bool) -> plt.Figure:
    """保存或显示图形并关闭，返回 figure 以便进一步处理。"""
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)
    return fig


def plot_score_distribution(
    df: pd.DataFrame, save_path: Optional[str] = None, show: bool = False
) -> plt.Figure:
    """标签 score 取值分布（计数 + 占比标注）。"""
    _non_empty(df)
    dist = score_distribution(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=dist, x="score", y="count", ax=ax, color="#4C72B0")
    for _, row in dist.iterrows():
        ax.text(float(row["score"]), float(row["count"]), f"{float(row['proportion']):.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Score Distribution (label imbalance)")
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    return _finalize(fig, save_path, show)


def plot_context_distribution(
    df: pd.DataFrame, top_n: int = 20, save_path: Optional[str] = None, show: bool = False
) -> plt.Figure:
    """Top-N context（CPC 分类）样本数分布。"""
    _non_empty(df)
    dist = context_distribution(df, top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.32)))
    sns.barplot(data=dist, x="count", y="context", ax=ax, color="#55A868")
    ax.set_title(f"Top-{top_n} Context (CPC class) by sample count")
    ax.set_xlabel("count")
    ax.set_ylabel("context")
    return _finalize(fig, save_path, show)


def plot_score_by_context(
    df: pd.DataFrame, top_n: int = 20, save_path: Optional[str] = None, show: bool = False
) -> plt.Figure:
    """Top-N context 的平均 score（按样本数排序），用于比较各领域标注难度。"""
    _non_empty(df)
    sbc = score_by_context(df).head(top_n).sort_values("mean_score")
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.32)))
    sns.barplot(data=sbc, x="mean_score", y=sbc.index, ax=ax, color="#C44E52")
    ax.set_title(f"Mean score by Top-{top_n} context")
    ax.set_xlabel("mean score")
    ax.set_ylabel("context")
    return _finalize(fig, save_path, show)


def plot_length_distributions(
    df: pd.DataFrame, save_path: Optional[str] = None, show: bool = False
) -> plt.Figure:
    """anchor / target 的字符长度分布直方图（用于确定 max_seq_len）。"""
    _non_empty(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, color in zip(axes, ("anchor", "target"), ("#4C72B0", "#DD8452")):
        lens = df[col].astype(str).str.len()
        sns.histplot(lens, ax=ax, bins=40, color=color, kde=False)
        ax.axvline(lens.quantile(0.95), color="red", linestyle="--", label="p95")
        ax.set_title(f"{col} length (chars)")
        ax.set_xlabel("char length")
        ax.set_ylabel("count")
        ax.legend()
    fig.tight_layout()
    return _finalize(fig, save_path, show)


def plot_token_overlap(
    df: pd.DataFrame, save_path: Optional[str] = None, show: bool = False
) -> plt.Figure:
    """不同 score 下 anchor↔target 的词重叠度（Jaccard）均值对比。"""
    _non_empty(df)
    to = token_overlap(df).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=to, x="score", y="mean", ax=ax, color="#8172B3")
    for _, row in to.iterrows():
        ax.text(float(row["score"]), float(row["mean"]), f"{float(row['mean']):.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Mean token-overlap (Jaccard) by score")
    ax.set_xlabel("score")
    ax.set_ylabel("mean Jaccard")
    return _finalize(fig, save_path, show)


def plot_all(df: pd.DataFrame, out_dir: str = "figures") -> list[str]:
    """一次性生成全部可视化并保存到 ``out_dir``，返回生成的图片路径列表。

    这是最省事的「可视化操作」入口：本地运行后直接在 out_dir 打开图片，
    或上传到 Kaggle 后在 notebook 中查看。
    """
    _non_empty(df)
    names = [
        "score_distribution",
        "context_distribution",
        "score_by_context",
        "length_distributions",
        "token_overlap",
    ]
    paths: list[str] = []
    for name in names:
        save_path = os.path.join(out_dir, f"{name}.png")
        globals()[f"plot_{name}"](df, save_path=save_path, show=False)
        paths.append(save_path)
    return paths
