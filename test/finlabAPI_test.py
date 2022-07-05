from finlab import data
import finlab

finlab.login("tiyybDDkjfj91vIexZCvLwL84ClFRIKE2yQwZBKP/dpT8FMnMdr9dxTKxwt29KjD#free")

# lstInvestorsInfo = ["institutional_investors_trading_summary:投信買賣超股數"]
# dfFinancialStatement = data.get('institutional_investors_trading_summary:投信買賣超股數')
# pb = data.get('price_earning_ratio:股價淨值比')

# ========================================================

close_index = data.get('stock_index_price:收盤指數')
print(f"close_index.columns: {list(close_index.columns)}")
# print(close_index.head())
# print()
# print(close_index.tail())

# ========================================================

# company_basic_info = data.get('company_basic_info')
# print(f"company_basic_info.columns: {company_basic_info.columns}")
# company_basic_info = company_basic_info.sort_values(by=['實收資本額(元)'], ascending=False)

# company_basic_info = company_basic_info[company_basic_info["產業類別"] == "鋼鐵工業"][["stock_id", "公司名稱", "實收資本額(元)"]]
# print(f"len(company_basic_info): {len(company_basic_info)}")
# print(company_basic_info.head())
# print()
# print(company_basic_info.tail())

# ========================================================

# close = data.get('price:收盤價')
# print(close.head())
# print()
# print(close.tail())

# print(type(close))