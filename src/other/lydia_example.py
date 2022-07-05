import pandas as pd

df = pd.DataFrame(data={
    'article' : ['....台積電...','....廣達....%^%^$%.','...鴻海$#%^#', '$%&$$%**^']
})

def process(article):
    subString_list = ["台積電", "廣達", "鴻海"]     # 把這裡換成妳想保留的公司名
    for subString in subString_list:
        if subString in article:
            return subString
    return None    # 如果找不到就回傳None，後面再把有None的row drop掉

df["companyName"] = df["article"].apply(process)
print(df)