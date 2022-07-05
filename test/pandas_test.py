# from email.utils import parsedate_to_datetime
import pandas as pd

# df1 = pd.DataFrame({"a":[1,2,3], "b": [4,5,6]})
# df2 = pd.DataFrame({"c":[7,8,9], "d": [10,11,12]})
# df3 = pd.DataFrame({"a":[7,8,9], "b": [10,11,12]})

# df = pd.concat([df1, df2], ignore_index=True, sort=False)
# print(df.head())

# df = pd.concat([df1, df3], ignore_index=True, sort=False)
# print(df.head())

# ===============================

# df1 = pd.DataFrame({"stockID":[1101,1101,1101, 1102, 1102, 1102], 
#                     "date": ["2021/01/01", "2021/01/02", "2021/01/03", "2021/01/01", "2021/01/02", "2021/01/03"],
#                     "price": [10, 11, 12, 20, 21, 22]
# })
# print(df1)

# df1["date"] = pd.to_datetime(df1["date"])
# print(df1)

# df1.set_index(['stockID', 'date'], inplace=True)
# print(df1)

# # print(df1[1101])  # (X)
# # print(df1.index.get_level_values(0))
# print(df1.xs(1101, level='stockID'))
# print(df1.loc(1101))

# ==============================

df = pd.DataFrame({"item":["open", "open", "open", "close", "close", "close"], 
                    "date": ["2021/01/01", "2021/01/02", "2021/01/03", "2021/01/01", "2021/01/02", "2021/01/03"],
                    "1101": [10, 11, 12, 20, 21, 22],
                    "1102": [9, 10, 11, 19, 20, 21],
})
print(df)

df = df.pivot(index="date", columns="item")
print(df)

print(df.columns.get_level_values(0).unique())

# df1 = df["1101"]
# df1["stockID"] = 1101
# df1 = df1.reset_index()
# df1 = df1.set_index(["stockID", "date"])
# print(df1)

# df2 = df["1102"]
# df2["stockID"] = 1102
# df2 = df2.reset_index()
# df2 = df2.set_index(["stockID", "date"])
# print(df2)

# df3 = pd.concat([df1, df2], sort=False)
# print(df3)

lstDFs = []
for stockID in df.columns.get_level_values(0).unique():
    dfTmp = df[stockID]
    dfTmp["stockID"] = int(stockID)
    dfTmp = dfTmp.reset_index()
    dfTmp = dfTmp.set_index(["stockID", "date"])
    lstDFs.append(dfTmp)

df4 = pd.concat(lstDFs, sort=False)
print(df4)

print(df4.loc[1101])