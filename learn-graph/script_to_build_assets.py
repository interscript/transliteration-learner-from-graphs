

from __future__ import unicode_literals
from hazm import *

import pandas as pd
import numpy as np
import openpyxl
import _pickle as cPickle


# Ressources expected: Flexicon/Affixes.xlsx  Flexicon/Entries.xlsx
PATH_FLEXICON = "resources/"
# Tagger model
PATH_HAZM = PATH_FLEXICON+"postagger.model"
# PoS mappings
PATH_MAPPINGS = PATH_FLEXICON+"PoS_mappings.csv"


stemmer = Stemmer()
lemmatizer = Lemmatizer()
normalizer = Normalizer()
tagger = POSTagger(model=PATH_HAZM)


idces_Entries = ["PhonologicalForm", "WrittenForm", "SynCatCode", "Freq", "الگوی تکیه"]

path = PATH_FLEXICON + "Entries.xlsx"
wb_obj = openpyxl.load_workbook(path)
sheet_obj = wb_obj.active

df_Entries = pd.DataFrame(sheet_obj.values)
df_Entries = df_Entries.set_axis(idces_Entries, axis=1)[1:]

d_df_Entries = dict(list(df_Entries.groupby("SynCatCode")))

idces_Affixes = [
    "id",
    "Affix",
    "PhonologicalForm",
    "نام وند",
    "کد معنا",
    "جایگاه",
    "نوع",
    "طرح تکیه",
    "واکه",
    "درج واج در وندافزایی",
    "تغییرات آوایی در وندافزایی",
    "SynCatCode",
    "مقولة ستاک+وند",
    "هم\u200cنویسه با واحد واژگانی",
    "هجاسازی مجدد",
]


path = PATH_FLEXICON + "Affixes.xlsx"
wb_obj = openpyxl.load_workbook(path)
sheet_obj = wb_obj.active

df_Affixes = pd.DataFrame(sheet_obj.values)
df_Affixes = df_Affixes.set_axis(idces_Affixes, axis=1)[1:]

# mappings
df = pd.read_csv(PATH_MAPPINGS)

d_FLEXI = {}
for i in range(df.shape[0]):
    d = df[["PoS", "FLEXICON"]].iloc[i].to_dict()
    d_FLEXI[d["PoS"]] = d["FLEXICON"]

d_df_FLEXI = {}
for k in d_FLEXI.keys():
    dims = d_FLEXI[k]
    if type(dims) == str:
        d_df_FLEXI[k] = pd.concat(
            [df_Entries[df_Entries["SynCatCode"] == d] for d in dims.split(",")]
        )

d_map_FLEXI = {}
for k in d_FLEXI.keys():
    dims = d_FLEXI[k]
    print(dims)
    if type(dims) == str:
        for d in dims.split(","):
            if d.strip() != "":
                d_map_FLEXI[d.strip()] = k

d_HAZM = {}
for i in range(df.shape[0]):
    d = df[["PoS", "Hazm"]].iloc[i].to_dict()
    d_HAZM[d["PoS"]] = d["Hazm"]


d_map_HAZM = {}
for k in d_HAZM.keys():
    dims = d_HAZM[k]
    # print(dims)
    if type(dims) == str:
        for d in dims.split(","):
            if d.strip() != "":
                d_map_HAZM[d.strip()] = k


dic_assets = {
    'affixes': df_Affixes,
    'entries': df_Entries,
    'd_FLEXI': d_FLEXI,
    'd_map_FLEXI': d_map_FLEXI,
    'd_HAZM': d_HAZM,
    'd_map_HAZM': d_map_HAZM
}

import pickle

with open('resources/farsi_assets.pickle', 'wb') as handle:
    pickle.dump(dic_assets, handle, protocol=pickle.HIGHEST_PROTOCOL)
