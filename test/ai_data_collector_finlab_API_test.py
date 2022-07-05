from finlab import data
import finlab
import pandas as pd

finlab.login("tiyybDDkjfj91vIexZCvLwL84ClFRIKE2yQwZBKP/dpT8FMnMdr9dxTKxwt29KjD#free")


# dctPriceItems = {"開盤價":"open","收盤價":"close","最高價":"high","最低價":"low","成交股數":"volume_stocks","成交筆數":"volume_deals"}
# df = pd.DataFrame()
# for k,v in dctPriceItems.items():
#     print(f"processing {k}...")
#     dfPrice = data.get(f'price:{k}')
#     #   df[v] = dfPrice
#     df = pd.concat([df, dfPrice], ignore_index=True, sort=False)

# print(df.head())

# all_investor_summary_df = data.get('institutional_investors_trading_all_market_summary:買賣超').rank(pct=True, axis=1)
all_investor_summary_df = data.get('institutional_investors_trading_all_market_summary:買賣超')
print(all_investor_summary_df.head())
print()
print(all_investor_summary_df.tail())

# foreign_investor_without_dealer_df = data.get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)').rank(pct=True, axis=1)
# foreign_investor_without_dealer_df = data.get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')
# print(foreign_investor_without_dealer_df.head())
# print()
# print(foreign_investor_without_dealer_df.tail())

# foreign_investor = data.get('institutional_investors_trading_summary:外資自營商買賣超股數').rank(pct=True, axis=1)
# investor =  data.get('institutional_investors_trading_summary:投信買賣超股數').rank(pct=True, axis=1)
# dealer_internal = data.get('institutional_investors_trading_summary:自營商買賣超股數(自行買賣)').rank(pct=True, axis=1)
# dealer_hedging = data.get('institutional_investors_trading_summary:自營商買賣超股數(避險)').rank(pct=True, axis=1)
