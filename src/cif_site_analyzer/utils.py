import re
import importlib.resources
import pandas as pd
import matplotlib.colors as mcolors
from collections import defaultdict

with importlib.resources.as_file(
    importlib.resources.files("cif_site_analyzer.data")
    / "elemental-property-list.csv"
) as csv_path:
    features = pd.read_csv(csv_path)

all_elements = features["Symbol"].tolist()
MNs = dict(zip(all_elements, features["Mendeleev number"].tolist()))

# add actinides for handling prototypes with elements for which
# features are not available
for e, mn in [["Ac", 14], ["Pa", 18], ["Np", 22], ["Pu", 24], ["Am", 26]]:
    MNs[e] = mn


def get_colors(colors, name_map):
    print("The colors for the different groups are: ")
    for i, (group, color) in enumerate(zip(name_map.keys(), colors), 1):
        print(f"{i}. {group:<5} : {color}")

    res = get_valid_input(
        "Enter y to accept the assigned colors or n to customize : ",
        ["Y", "y", "N", "n"],
    )[0]
    if res == "n":
        prompt = f"Enter colors for groups 1..{len(name_map)}\
            , separated by comma : "
        colors = get_valid_input(
            prompt,
            mcolors.CSS4_COLORS,
        )

    return colors


def get_colormap(color, N=256):

    color_list = [(0.0, "white"), (1.0, color)]
    cmap = None
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_purple_green_orange_red", [t[1] for t in color_list], N=N
    )

    return cmap


def sort_group_labesl_by_MN(df, site_assigment):

    group_MN = defaultdict(list)
    for _, row in df.iterrows():
        for group, sites in site_assigment.items():
            for site in sites:
                pform = _parse_formula(row[site])
                for k, _ in pform.items():
                    if k in all_elements:
                        group_MN[group].append(MNs[k])

    average_MNs = []
    for group, mns in group_MN.items():
        average_MNs.append([group, sum(mns) / len(mns)])

    average_MNs = sorted(average_MNs, key=lambda x: x[1])
    group_rename_map = dict(
        zip(
            [k[0] for k in average_MNs],
            [f"G{i}" for i in range(1, len(average_MNs) + 1)],
        )
    )

    avg_MNs = dict(
        zip(
            [f"G{i}" for i in range(1, len(average_MNs) + 1)],
            [k[1] for k in average_MNs],
        )
    )

    ptype = _parse_formula(df.iloc[0]["Entry prototype"].split(",")[0])
    ptype = sorted([[k, MNs[k]] for k, _ in ptype.items()], key=lambda r: r[1])
    name_map = dict(zip(avg_MNs.keys(), [p[0] for p in ptype]))

    return group_rename_map, avg_MNs, name_map


def concat_site_formula(row, site_assignment):
    group_formulae = []
    for _, sites in site_assignment.items():
        sites = list(sites)
        groupf = row[sites[0]]
        for site in sites[1:]:
            groupf += row[site]

        groupf = _parse_formula(groupf)

        groups = ""
        for e, c in groupf.items():
            if c == 1.0:
                groups += e
            elif abs(c - int(c)) == 0:
                groups += f"{e}{int(c)}"
            else:
                groups += f"{e}{round(float(c), 2)}"
        group_formulae.append(groups)

    return group_formulae


def auto_group_sites(wyckoff_symbol_elements, stype):

    elem_sites = _parse_formula(stype.split(",")[0]).keys()

    wyckoff_symbol_elements_s = []
    for ws, site_formula in wyckoff_symbol_elements.items():
        elements = set()
        for form in site_formula:
            form = sorted(
                [[k, v] for k, v in _parse_formula(form).items()],
                key=lambda x: x[1],
            )
            elements.add(form[-1][0])
        wyckoff_symbol_elements_s.append([ws, elements])

    threshold = 0.1
    delta = 0.05
    n_try = 0
    converged = False
    while not converged:
        groups = {}
        grouped_sites = {}
        gc = 1
        for i in range(len(wyckoff_symbol_elements_s)):
            ws1, el1 = wyckoff_symbol_elements_s[i]
            for j in range(i + 1, len(wyckoff_symbol_elements_s)):
                ws2, el2 = wyckoff_symbol_elements_s[j]

                dis_score = len(el1.union(el2) - el1.intersection(el2)) / max(
                    len(el1), len(el2)
                )

                if dis_score <= threshold:
                    if ws1 in grouped_sites:
                        groups[grouped_sites.get(ws1)].add(ws2)
                        grouped_sites[ws2] = grouped_sites.get(ws1)
                    else:
                        groups[f"G{gc}"] = set([ws1, ws2])
                        grouped_sites[ws1] = f"G{gc}"
                        grouped_sites[ws2] = f"G{gc}"
                        gc += 1
            if ws1 not in grouped_sites:
                groups[f"G{gc}"] = set([ws1])
                gc += 1
            else:
                grouped_sites[ws1] = grouped_sites.get(ws1)

        if len(elem_sites) == len(groups):
            converged = True
        else:
            threshold += delta

        n_try += 1
        if n_try > 18 and delta != 0.01:
            threshold = 0.01
            delta = 0.01

        if n_try > 120:
            print(
                "Threshold value for group assignment did not converge."
                "Assign the groups manually."
            )
            return groups

    return groups


def list_to_formula(values):
    form = defaultdict(float)
    for v in values:
        for e, c in _parse_formula(v).items():
            form[e] += c

    sform = ""
    for e, c in form.items():
        if c == 1.0:
            sform += f"{e}"
        elif abs(c - int(c)) < 1e-3:
            sform += f"{e}{int(c)}"
        else:
            sform += f"{e}{round(c, 2)}"

    return sform


def assign_labels_for_sites(wyckoff_symbols, site_assignment=None):
    wyckoff_symbols = list(wyckoff_symbols)
    print("\nThe following sites are present in the current dataset.")
    for i, wsymbol in enumerate(wyckoff_symbols, 1):
        print(f"{i}. {wsymbol}")

    if site_assignment is None:
        site_assignment = {
            f"S{i}": [ws] for i, ws in enumerate(wyckoff_symbols, 1)
        }
    print(f"\nThe automatic site assignment is {site_assignment}.")

    result = get_valid_input(
        "Press y to accept or n to customize: ", ["Y", "y", "N", "n"]
    )[0].lower()

    if result == "n":
        new_assigmnet = {}
        while len(wyckoff_symbols):
            print(f"Available sites: {wyckoff_symbols}")
            wys = get_valid_input(
                "\nEnter group label (e.g. R, M, X): ", None
            )[0]

            wsites = get_valid_input(
                f"Enter Wyckoff symbols from {', '.join(wyckoff_symbols)}: ",
                wyckoff_symbols,
            )

            new_assigmnet[wys] = wsites

            for ws in wsites:
                del wyckoff_symbols[wyckoff_symbols.index(ws)]
        site_assignment = new_assigmnet

    print(f"\nCurrent group assignments are {site_assignment}\n")
    return site_assignment


def get_valid_input(prompt, valid_choices=None):
    while True:
        user_input = input(prompt)
        answers = [x.strip() for x in user_input.split(",")]
        if valid_choices:
            if all(ans in valid_choices for ans in answers):
                return answers
            else:
                msg = "Invalid input. Please enter only values from: "
                msg += f"{', '.join(valid_choices)} (comma-separated)"
                print(msg)
        else:
            return answers


def get_ptable_vals_dict(values):
    # print(values[:5])
    table_values = defaultdict(float)
    for form in values:

        pform = _parse_formula(form)
        for k, v in pform.items():
            table_values[k] += v
    return table_values


def get_validated_input(prompt, valid_values):
    """
    Prompt the user for input: main items (comma-separated),
    optional items after "ex:". Validates each against
    provided lists and returns dictionary with results.
    """
    while True:
        user_input = input(prompt).strip()
        # Split into main and optional by "ex:"
        parts = user_input.split("ex:")
        inlcude_part = parts[0].strip()
        exclude_part = parts[1].strip() if len(parts) > 1 else ""

        main_items = [
            item.strip() for item in inlcude_part.split(",") if item.strip()
        ]
        optional_items = [
            item.strip() for item in exclude_part.split(",") if item.strip()
        ]

        # Validate main and optional items
        invalid_include = [
            item for item in main_items if item not in valid_values
        ]
        invalid_exclude = [
            item for item in optional_items if item not in valid_values
        ]

        if not invalid_include and not invalid_exclude:
            return {"include": main_items, "exclude": optional_items}
        print("Invalid input detected.")
        if invalid_include:
            msg = "Invalid selection. Please select from : "
            msg += f"{', '.join(invalid_include)}"
        if invalid_exclude:
            msg = "Invalid selection. Please select from : "
            msg += f"{', '.join(invalid_exclude)}"
        print(msg)
        print("Please try again.\n")


def get_selected_elements(site_assignment):

    features = pd.read_csv("data/elemental-property-list.csv")

    def get_gp(symbol):
        f4 = [
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
        ]
        f5 = ["Th", "U"]
        period_map = {
            1: "1P",
            2: "2P",
            3: "3P",
            4: "4P",
            5: "5P",
            6: "6P",
            7: "7P",
        }

        group_map = {
            1: "1A",
            2: "2A",
            3: "3B",
            4: "4B",
            5: "5B",
            6: "6B",
            7: "7B",
            8: "8B",
            9: "8B",
            10: "8B",
            11: "1B",
            12: "2B",
            13: "3A",
            14: "4A",
            15: "5A",
            16: "6A",
            17: "7A",
            18: "8A",
        }

        if symbol in f4:
            period = "4f"
        elif symbol in f5:
            period = "5f"
        else:
            period = period_map.get(
                features[features["Symbol"] == symbol]["Period"].values[0]
            )

        if symbol in f4 or symbol in f5:
            group = "3r"
        else:
            group = group_map.get(
                features[features["Symbol"] == symbol]["Group"].values[0]
            )

        return period, group

    features[["P", "G"]] = features.apply(
        lambda r: get_gp(r["Symbol"]), result_type="expand", axis=1
    )

    periods = [
        "1P",
        "2P",
        "3P",
        "4P",
        "5P",
        "6P",
        "7P",
        "4f",
        "5f",
        "1p",
        "2p",
        "3p",
        "4p",
        "5p",
        "6p",
        "7p",
        "4F",
        "5F",
    ]
    groups = [
        "1A",
        "2A",
        "3B",
        "4B",
        "5B",
        "6B",
        "7B",
        "8B",
        "1B",
        "2B",
        "3A",
        "4A",
        "5A",
        "6A",
        "7A",
        "1a",
        "2a",
        "3b",
        "4b",
        "5b",
        "6b",
        "7b",
        "8b",
        "1b",
        "2b",
        "3a",
        "4a",
        "5a",
        "6a",
        "7a",
    ]

    valid_inputs = (
        periods + groups + features["Symbol"].tolist() + ["Ae", "ae"]
    )

    selection_for_labels = {}

    print("\nPlease select the elements for candidate screening.")
    print(
        "Type ae to select all elements, or choose from the \
            following groups and periods."
    )
    msg = f"\tPeriods : {', '.join(periods[:int(len(periods)/2)])} "
    msg += f"\n\tGroups  : {', '.join(groups[:int(len(groups)/2)])}"
    print(msg)
    print("Exclusions can be added following the ex: tag.")
    print(
        "For example, to select alkali and alkaline earth \
            metals excluding Rb and Sr, type 1A,2A ex:Rb,Sr\n"
    )
    for site, wys in site_assignment.items():
        selection = get_validated_input(
            f"Enter selection for {site} ({', '.join(wys)}): ", valid_inputs
        )

        if selection["include"][0].lower() == "ae":
            elements = features["Symbol"].tolist()
        else:
            elements = []
            for gp in selection["include"]:
                gp = str(gp[0]) + gp[-1].upper()
                if gp[-1] == "F":
                    gp = str(gp[0]) + "f"
                elements.extend(
                    features[features["P"] == gp]["Symbol"].tolist()
                )
                elements.extend(
                    features[features["G"] == gp]["Symbol"].tolist()
                )
        if len(selection["exclude"]):
            elements = [e for e in elements if e not in selection["exclude"]]

        selection_for_labels[site] = elements

    return selection_for_labels


def _parse_formula(formula: str, strict: bool = True) -> dict[str, float]:
    """Copied from pymatgen.

    Args:
        formula (str): A string formula, e.g. Fe2O3, Li3Fe2(PO4)3.
        strict (bool): Whether to throw an error if formula
        string is invalid (e.g. empty).
            Defaults to True.

    Returns:
        Composition with that formula.

    Notes:
        In the case of Metallofullerene formula (e.g. Y3N@C80),
        the @ mark will be dropped and passed to parser.
    """
    # Raise error if formula contains special characters
    # or only spaces and/or numbers
    for char in ["'", "-", "+", "-"]:
        formula = formula.replace(char, "")

    if strict and re.match(r"[\s\d.*/]*$", formula):
        print(formula)
        raise ValueError(f"Invalid {formula=}")

    # For Metallofullerene like "Y3N@C80"
    formula = formula.replace("@", "")
    # Square brackets are used in formulas to denote
    # coordination complexes (gh-3583)
    formula = formula.replace("[", "(")
    formula = formula.replace("]", ")")

    def get_sym_dict(form: str, factor: float) -> dict[str, float]:
        sym_dict: dict[str, float] = defaultdict(float)
        for match in re.finditer(r"([A-Z][a-z]*)\s*([-*\.e\d]*)", form):
            el = match[1]
            amt = 1.0
            if match[2].strip() != "":
                amt = float(match[2])
            sym_dict[el] += amt * factor
            form = form.replace(match.group(), "", 1)
        if form.strip():
            raise ValueError(f"{form} is an invalid formula!")
        return sym_dict

    match = re.search(r"\(([^\(\)]+)\)\s*([\.e\d]*)", formula)
    while match:
        factor = 1.0
        if match[2] != "":
            factor = float(match[2])
        unit_sym_dict = get_sym_dict(match[1], factor)
        expanded_sym = "".join(
            f"{el}{amt}" for el, amt in unit_sym_dict.items()
        )
        expanded_formula = formula.replace(match.group(), expanded_sym, 1)
        formula = expanded_formula
        match = re.search(r"\(([^\(\)]+)\)\s*([\.e\d]*)", formula)
    return get_sym_dict(formula, 1)
