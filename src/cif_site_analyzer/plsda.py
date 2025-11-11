import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cif_site_analyzer.config import FEATURE_GROUPS_PLSDA
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from collections import defaultdict
import matplotlib.patches as patches


def plot_confidence_ellipse(
    x,
    y,
    ax,
    n_std=2.0,
    edgecolor="black",
    facecolor="none",
    alpha=1.0,
    **kwargs,
):
    if x.size != y.size:
        raise ValueError("x and y must have the same size")

    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    eigvals, eigvecs = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)

    ellipse = Ellipse(
        (mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        edgecolor=edgecolor,
        facecolor=facecolor,
        lw=2,
        alpha=alpha,
        **kwargs,
    )
    ax.add_patch(ellipse)
    return ellipse


def run_pls_da(
    df, color_map, output_loadings_excel="outputs/csv/PLS_DA_loadings.csv"
):
    # -----------------------------
    # Prepare the data for PLS-DA
    # -----------------------------
    # Drop non-numerical columns: "Formula", "Site", "Site_Label"
    X = df.drop(columns=["Formula", "Site_formula", "Site_label"])
    y = df["Site_label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    minmax_scaler = MinMaxScaler()
    X_normalized = minmax_scaler.fit_transform(X_scaled)

    y_dummies = pd.get_dummies(y)
    Y = y_dummies.values  # shape: (n_samples, n_classes)

    # -------------------------------------
    # Fit the PLS-DA model with 2 components
    # -------------------------------------
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_normalized, Y)

    X_scores = pls.x_scores_

    # ---------------------------------------
    # Compute explained variance (for labels)
    # ---------------------------------------
    total_variance = np.sum(np.var(X_normalized, axis=0))
    explained_variances = np.var(X_scores, axis=0)
    explained_ratio = explained_variances / total_variance * 100

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)

    unique_classes = y.unique()

    for cls in unique_classes:
        idx = y == cls
        points = X_scores[idx, :]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            color=color_map[cls],
            label=cls,
            s=250,
            alpha=0.5,
            edgecolors="none",
        )
        plot_confidence_ellipse(
            points[:, 0],
            points[:, 1],
            ax,
            n_std=2.0,
            edgecolor=color_map[cls],
            facecolor=color_map[cls],
            alpha=0.2,
        )

    ax.set_xlabel(f"LV1 ({explained_ratio[0]:.1f}%)", fontsize=18)
    ax.set_ylabel(f"LV2 ({explained_ratio[1]:.1f}%)", fontsize=18)

    legend = ax.legend(fontsize=18)
    for text in legend.get_texts():
        text.set_fontstyle("italic")

    plt.tick_params(axis="both", labelsize=18)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/PLS_DA_Scatter_Plot.svg", dpi=500)
    # plt.show()

    # --------------------------------------
    # Compute and output full loadings per component
    # --------------------------------------
    x_loadings = pls.x_loadings_  # shape: (n_features, 2)
    loadings_df = pd.DataFrame(
        x_loadings,
        columns=[f"Component_{i+1}_Loading" for i in range(pls.n_components)],
    )
    loadings_df.insert(0, "Feature", X.columns.tolist())

    os.makedirs(os.path.dirname(output_loadings_excel), exist_ok=True)
    loadings_df.to_csv(output_loadings_excel, index=False)
    print(f"Full loadings saved to {output_loadings_excel}.")

    # ------------------------------------------------
    # Create top_contribution.xlsx listing only top 10
    # ------------------------------------------------
    # For each component, sort features by absolute loading (descending)
    comp1_loadings = loadings_df["Component_1_Loading"].abs()
    comp2_loadings = loadings_df["Component_2_Loading"].abs()

    sorted_feats_comp1 = loadings_df.loc[
        comp1_loadings.sort_values(ascending=False).index, "Feature"
    ].tolist()[
        :10
    ]  # only top 10
    sorted_feats_comp2 = loadings_df.loc[
        comp2_loadings.sort_values(ascending=False).index, "Feature"
    ].tolist()[
        :10
    ]  # only top 10

    # Build a DataFrame whose columns are the
    # top 10 features for each component
    header1 = f"Component 1 ({explained_ratio[0]:.1f}%)"
    header2 = f"Component 2 ({explained_ratio[1]:.1f}%)"

    top_contrib_df = pd.DataFrame(
        {
            header1: pd.Series(sorted_feats_comp1),
            header2: pd.Series(sorted_feats_comp2),
        }
    )

    top_contrib_path = os.path.join("outputs", "csv", "top_contribution.csv")
    top_contrib_df.to_csv(top_contrib_path, index=False)
    print(f"Top 10 features saved to {top_contrib_path}.")

    plot_loadings(loadings_df, explained_ratio)

    return loadings_df, scaler, minmax_scaler


def plot_loadings(loadings_df, explained_ratio):

    colors = {
        "Themal & Physical Properties": "tab:green",
        "Electronegativity & Electron Affinity": "tab:olive",
        "DFT RLDA & ScRLDA Properties": "tab:brown",
        "Basic Atomic Properties": "tab:red",
        "DFT LDA & LSD Properties": "tab:purple",
        "Atomic & Ionic Radii": "tab:blue",
        "Valence Properties": "tab:orange",
    }

    plt.clf()
    plt.close("all")
    plt.style.use("default")

    fig, ax = plt.subplots()

    xr = (
        loadings_df["Component_1_Loading"].max()
        - loadings_df["Component_1_Loading"].min()
    )
    yr = (
        loadings_df["Component_2_Loading"].max()
        - loadings_df["Component_2_Loading"].min()
    )

    added_legends = []
    group_points = defaultdict(list)
    for i, (_, row) in enumerate(loadings_df.iterrows(), 1):
        gname = FEATURE_GROUPS_PLSDA[row["Feature"]]

        group_points[gname].append(
            [row["Component_1_Loading"], row["Component_2_Loading"]]
        )
        if gname in added_legends:
            ax.scatter(
                row["Component_1_Loading"],
                row["Component_2_Loading"],
                color=colors[gname],
                s=150,
                alpha=0.5,
            )
        else:
            ax.scatter(
                row["Component_1_Loading"],
                row["Component_2_Loading"],
                color=colors[gname],
                s=150,
                alpha=0.5,
                label=gname,
            )

            added_legends.append(gname)

        ax.text(
            row["Component_1_Loading"] - xr * 0.015,
            row["Component_2_Loading"] - yr * 0.013,
            s=f"{i:02}",
            fontsize=8,
        )

    for gname, vals in group_points.items():
        vals = np.array(vals)
        center = np.mean(vals, axis=0).tolist()

        cov = np.cov(vals[:, 0], vals[:, 1])
        vals, vecs = np.linalg.eigh(cov)
        height, width = np.sqrt(5.991) * np.sqrt(vals)
        angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))

        ellipse = patches.Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle,
            edgecolor="none",
            facecolor=colors[gname],
            alpha=0.2,
        )

        ax.add_patch(ellipse)

    plt.xlabel(f"LV1 ({explained_ratio[0]:.1f} %)")
    plt.ylabel(f"LV2 ({explained_ratio[1]:.1f} %)")
    plt.grid(visible=False)

    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        bbox_transform=plt.gcf().transFigure,
        ncol=2,
        frameon=True,
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig("outputs/plots/Feature_loadings.svg", bbox_inches="tight")
