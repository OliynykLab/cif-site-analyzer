from .get_data import load_cif
from .get_data import select_one_stype
from .get_data import add_average_coordinates
from .get_data import prepare_data_for_engine
from .utils import get_ptable_vals_dict
from .utils import list_to_formula
from .utils import assign_labels_for_sites
from .utils import auto_group_sites
from .utils import get_valid_input
from .utils import concat_site_formula
from .utils import sort_group_labesl_by_MN
from .utils import get_colors
from .utilities import merge_images
from .features import add_features
from .ptable_histogram import ptable_heatmap_mpl
from .plsda import run_pls_da
from .visualization import visualize_elements
from .projection import plot_elements_from_plsda_loadings
import pandas as pd
import importlib.resources
import argparse
import os


# Plot site heatmaps
cmaps = [
    "Reds",
    "Blues",
    "Greens",
    "Purples",
    "Oranges",
    "YlOrBr",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
]

color_dict = {
    2: ["blue", "red"],
    3: ["blue", "grey", "red"],
    4: ["blue", "grey", "green", "red"],
    5: ["blue", "purple", "green", "orange", "red"],
}


def main():
    description = """
        CIF Site Analyzer

        """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_path", help="Path to the input file / folder")
    parser.add_argument(
        "-e", "--elements", help="Must contain the specified element(s)"
    )
    parser.add_argument(
        "-s", "--font_size", help="Font size", default=40, type=int
    )
    parser.add_argument("-Y", help="Non Interactive", action="store_true")

    args = parser.parse_args()

    interactive = False if args.Y else True

    path = args.input_path
    elements = args.elements
    if elements:
        elements = elements.split(",")
        elements = set([e.strip() for e in elements])
    else:
        elements = set()

    if not os.path.isdir(path):
        print(f"Cannot find {path}.")
    path = os.path.join(os.getcwd(), path)
    os.chdir(path)

    font_size = args.font_size

    # DATA
    cif_data = load_cif(path, elements=elements, dev=False)
    stypes = select_one_stype(cif_data)

    if len(cif_data) == 0:
        print("No data found!")
        exit(0)

    if len(stypes) > 1:
        stypes = sorted(
            [[k, v] for k, v in stypes.items()], key=lambda x: x[1]
        )
        if interactive:
            print(
                "\n\nMore than one structure types are found in the input\
                      list. \nPlease select one structure type from \
                        the list below.\n"
            )

            for i, (stype, count) in enumerate(stypes, 1):
                print(f"({i}) {count:<5} {stype}")

            valid_input = False
            while not valid_input:
                res = input(
                    "Please enter the number corresponding to the \
                        selected structure type: \n"
                )
                try:
                    res = int(res)
                    assert 0 < res <= len(stypes)
                    selected_stype = stypes[res - 1][0]
                    valid_input = True
                except Exception:
                    # print(e)
                    print("Invalid input. Try again.")
        else:
            selected_stype = stypes[-1][0]
    else:
        selected_stype = list(stypes.keys())[0]

    data_for_engine, coordinate_data = prepare_data_for_engine(
        cif_data, selected_stype
    )
    os.makedirs("outputs/heatmaps", exist_ok=True)
    os.makedirs("outputs/heatmaps/individual", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/csv", exist_ok=True)

    # Save histograms
    data_df = pd.DataFrame(data_for_engine)
    wyckoff_symbols = data_df.columns[4:]

    # RMX assignment
    wyckoff_symbol_elements = {}
    for ws in wyckoff_symbols:
        wyckoff_symbol_elements[ws] = list(data_df[ws].unique())

    site_assignment = auto_group_sites(wyckoff_symbol_elements, selected_stype)

    if interactive:
        site_assignment = assign_labels_for_sites(
            wyckoff_symbols, site_assignment
        )

    data_df_w_groups = data_df.copy(deep=True)

    data_df_w_groups[list(site_assignment.keys())] = data_df_w_groups.apply(
        lambda r: concat_site_formula(r, site_assignment),
        axis=1,
        result_type="expand",
    )

    # sort according to average MN
    rename_map, avg_Mns, name_map = sort_group_labesl_by_MN(
        data_df_w_groups, site_assignment
    )
    data_df_w_groups.rename(columns=rename_map, inplace=True)

    site_assignment = {rename_map[k]: v for k, v in site_assignment.items()}
    site_assignment = sorted(
        [[k, v] for k, v in site_assignment.items()],
        key=lambda x: avg_Mns[x[0]],
        reverse=False,
    )
    site_assignment = {v[0]: v[1] for v in site_assignment}

    colors = color_dict.get(len(site_assignment))

    color_map = dict(zip(list(site_assignment.keys()), colors))

    # reassign
    if interactive:
        colors = get_colors(colors, name_map)

    # add average coordinates
    raw_data_output = add_average_coordinates(
        data_df_w_groups.copy(deep=True), coordinate_data
    )
    raw_data_output.to_csv(f"outputs/csv/{selected_stype}.csv", index=False)

    print("\nGenerating periodic table heatmaps...")
    for i, (k, v) in enumerate(site_assignment.items()):
        v = sorted(list(v))
        msg = f"Plotting periodic table heatmap for {k}: {name_map[k]} site"
        if len(v) > 1:
            msg += "s"
        print(msg)
        ptable_heatmap_mpl(
            vals_dict=get_ptable_vals_dict(data_df[v[0]].tolist()),
            site=name_map[k],
            stype=data_df.iloc[0]["Entry prototype"],
            cmap=color_map[k],
            group=k,
            font_size=font_size,
        )

    merge_images("outputs/heatmaps")

    for i, (k, v) in enumerate(site_assignment.items()):
        v = sorted(list(v))
        for _v in v:
            print(f"Plotting periodic table heatmap for {_v} site")
            ptable_heatmap_mpl(
                vals_dict=get_ptable_vals_dict(data_df[v[0]].tolist()),
                site=_v,
                stype=data_df.iloc[0]["Entry prototype"],
                cmap=color_map[k],
                group=k,
                font_size=font_size,
                individual=True,
            )
    print("Done, plots saved inside the directory plots.")

    # df for recommendation engine
    data = []
    for cif_data in data_for_engine:
        row = {
            "Filename": cif_data["Filename"],
            "Formula": cif_data["Formula"],
        }
        for group_label, sites in site_assignment.items():
            gcomp = []
            for site in sites:
                gcomp.append(cif_data[site])
            row[group_label] = list_to_formula(gcomp)
        data.append(row)

    df_engine = pd.DataFrame(data)
    df_engine["Notes"] = ""

    # add features
    print("\nLoading features...")
    features_df = add_features(df_engine)
    print("Done")

    # pls-da; 2-components
    print("\nPerforming PLS-DA...")
    pls_loadings, _, _ = run_pls_da(features_df, color_map)
    print("Done")

    # elements projection
    print("\nPlotting projections of elements and compounds...")
    with importlib.resources.as_file(
        importlib.resources.files("cif_site_analyzer.data")
        / "elemental-property-list.csv"
    ) as csv_path:
        coords = plot_elements_from_plsda_loadings(
            pls_loadings,
            color_map,
            df_engine,
            element_properties_file=csv_path,
        )
    visualize_elements(coords, df_engine, color_map, compounds_markers=False)
    visualize_elements(coords, df_engine, color_map, compounds_markers=True)
    print("Done, plots saved in the directory plots.")

    if interactive:

        print(df_engine.head())

        # get additional compounds for overlay
        def get_additional_cpds():
            res = get_valid_input(
                "\nDo you want to add compounds to compare ? ",
                ["Y", "y", "N", "n"],
            )[0]
            new_cpds = []

            if res.lower() == "y":
                print(
                    f"The current site-group assignment is {site_assignment}."
                )

                all_added = False
                nc = 1
                while not all_added:
                    new_cpd = get_valid_input(
                        "Enter the elements (without coefficients) at each "
                        "site-group for the new compound \
                            separated by comma : ",
                        None,
                    )
                    new_cpd = dict(zip(site_assignment.keys(), new_cpd))
                    new_cpd["Formula"] = "".join(list(new_cpd.values()))
                    new_cpd["Notes"] = "candidate"
                    new_cpd["Filename"] = f"NewCand_{nc}"
                    new_cpds.append(new_cpd)

                    done = get_valid_input(
                        "\nDo you wish to add more compounds ? ",
                        ["Y", "y", "N", "n"],
                    )[0]
                    if done.lower() == "n":
                        all_added = True
            return new_cpds

        new_cpds = get_additional_cpds()
        new_cpds = pd.DataFrame(new_cpds)

        df_engine = pd.concat([df_engine, new_cpds], axis=0, ignore_index=True)
        visualize_elements(
            coords, df_engine, color_map, compounds_markers=True
        )

        # elements for recommendations
        # elements_for_screening = get_selected_elements(site_assignment)

        # projection
        # recommendation_engine(site_element_pools=elements_for_screening,
        #                       sites_df=df_engine,
        #                       coord_df=coords,
        #                       output_file="outputs/recommendations.csv")

        # Sm4Ir2InGe4, Tb4Rh2InGe4, Lu4Ni2InGe4
