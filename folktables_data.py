import sys

import folktables as ft
import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.ensemble import RandomForestClassifier

import binary_confusion_matrix as bcm
from common import DEBUG

np.random.seed(0)

ACS_DATASETS = (
    ft.ACSEmployment,  # is employed is positive class. Rac1P protected group.
    ft.ACSHealthInsurance,
    ft.ACSPublicCoverage,
    ft.ACSTravelTime,
    ft.ACSMobility,
    ft.ACSEmploymentFiltered,
    ft.ACSIncomePovertyRatio
)

ORDINAL_MAP = {
    "COW": {
        1.0: (
            "Employee of a private for-profit company or"
            "business, or of an individual, for wages,"
            "salary, or commissions"
        ),
        2.0: (
            "Employee of a private not-for-profit, tax-exempt,"
            "or charitable organization"
        ),
        3.0: "Local government employee (city, county, etc.)",
        4.0: "State government employee",
        5.0: "Federal government employee",
        6.0: (
            "Self-employed in own not incorporated business,"
            "professional practice, or farm"
        ),
        7.0: (
            "Self-employed in own incorporated business,"
            "professional practice or farm"
        ),
        8.0: "Working without pay in family business or farm",
        9.0: "Unemployed and last worked 5 years ago or earlier or never worked",
    },
    "SCHL": {
        1.0: "No schooling completed",
        2.0: "Nursery school, preschool",
        3.0: "Kindergarten",
        4.0: "Grade 1",
        5.0: "Grade 2",
        6.0: "Grade 3",
        7.0: "Grade 4",
        8.0: "Grade 5",
        9.0: "Grade 6",
        10.0: "Grade 7",
        11.0: "Grade 8",
        12.0: "Grade 9",
        13.0: "Grade 10",
        14.0: "Grade 11",
        15.0: "12th grade - no diploma",
        16.0: "Regular high school diploma",
        17.0: "GED or alternative credential",
        18.0: "Some college, but less than 1 year",
        19.0: "1 or more years of college credit, no degree",
        20.0: "Associate's degree",
        21.0: "Bachelor's degree",
        22.0: "Master's degree",
        23.0: "Professional degree beyond a bachelor's degree",
        24.0: "Doctorate degree",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P":
    # Originally
    # {
    #     1.0: "White alone",
    #     2.0: "Black or African American alone",
    #     3.0: "American Indian alone",
    #     4.0: "Alaska Native alone",
    #     5.0: (
    #         "American Indian and Alaska Native tribes specified;"
    #         "or American Indian or Alaska Native,"
    #         "not specified and no other"
    #     ),
    #     6.0: "Asian alone",
    #     7.0: "Native Hawaiian and Other Pacific Islander alone",
    #     8.0: "Some Other Race alone",
    #     9.0: "Two or More Races",
    # },
        {
            1.0: "White",
            2.0: "Black",
            3.0: "Amer. Indian",
            4.0: "Alaska Native",
            5.0: "AI and AN",
            6.0: "Asian",
            7.0: "Pacific Islander",
            8.0: "Other",
            9.0: "Multiracial",
        }
}


def all_used_cols() -> set[str]:
    cols = set()
    for prob in ACS_DATASETS:
        cols.update(prob.features + [prob.target] + [prob.group])
    return cols


def get_data(states: list[str]):
    horizon = '1-Year'
    data_source = ft.ACSDataSource(survey_year='2017', horizon=horizon, survey='person')
    acs_data = data_source.get_data(states=states, download=True).drop_duplicates()
    keep_cols = all_used_cols()
    unseen_cols = keep_cols - set(acs_data.columns)
    if unseen_cols:
        print(f"Columns not found: {unseen_cols}", file=sys.stderr)
        return acs_data[list(keep_cols - unseen_cols)]

    return acs_data[list(keep_cols)]


def filter(
        data: pd.DataFrame,
        prob: ft.BasicProblem
) -> (pd.DataFrame, list[str]):
    cols = list(set(prob.features + [prob.target] + [prob.group]))
    for col in prob.features:
        if col not in data.columns:
            print(f"Column {col} not found in data.", file=sys.stderr)
            cols.remove(col)
    df = data[cols]
    df = df[df[prob.target].notna() & df[prob.group].notna()]
    df.dropna(inplace=True, axis=1)
    feat_cols = set(prob.features).intersection(df.columns)

    if "RAC1P" in df.columns:
        df['RAC1P'] = df['RAC1P'].map(ORDINAL_MAP['RAC1P'])
        dummies = pd.get_dummies(df["RAC1P"], drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        feat_cols = feat_cols.union(dummies.columns) - {"RAC1P"}

    if prob.target == "PINCP":
        df[prob.target] = df[prob.target] >= 50000

    feat_cols = list(feat_cols)
    df[feat_cols + [prob.target]] = df[feat_cols + [prob.target]].astype(float)

    if DEBUG:
        df = df.sample(frac=.1)

    return df, feat_cols


def get_preds(df: pd.DataFrame, feature_cols: list[str], target_col: str, group_col: str) -> pd.DataFrame:
    emp_train = df.sample(frac=.5)
    emp_test = df.drop(emp_train.index)

    print(f"Train size: {len(emp_train)}"
          f"\nTest size: {len(emp_test)}")
    print("Training...")
    clf = RandomForestClassifier()
    clf.fit(emp_train[feature_cols], emp_train[target_col])

    dfs = []
    for group, df in emp_test.groupby(group_col):
        dfs.append(pd.DataFrame({
            group_col: df[group_col],
            target_col: df[target_col],
            'pred': clf.predict(df[feature_cols])
        }))

    label_preds = pd.concat(dfs)

    return label_preds


def get_sample_labels() -> pd.DataFrame:
    df = get_data(states=["CA"])
    emp_df, feat_cols = filter(df, ft.ACSIncome)
    print(f"Columns: {emp_df.columns}")
    label_preds = get_preds(emp_df, feat_cols, ft.ACSIncome.target, ft.ACSIncome.group)
    return label_preds


def get_group_cm(label_preds: pd.DataFrame, group_name: str, normalize=False) -> bcm.BinaryConfusionMatrix:
    group_df = label_preds[label_preds[ft.ACSIncome.group] == group_name]
    cm = bcm.BinaryConfusionMatrix(
        skm.confusion_matrix(group_df[ft.ACSIncome.target], group_df['pred'], labels=[0, 1]),
        from_sklearn=True,
        normalize=normalize
    )
    return cm
