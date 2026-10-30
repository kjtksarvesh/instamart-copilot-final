from pathlib import Path
import sqlite3,pandas as pd
R=Path(__file__).resolve().parent
c=pd.read_csv(R/'data/clean/customers.csv');o=pd.read_csv(R/'data/clean/orders.csv')
assert len(c)==1200 and c.city.nunique()==6 and c.age.isna().sum()==0
assert len(o)==12309 and o.order_id.nunique()==12309 and not o.status.eq('pending').any()
assert int(o.status.eq('completed').sum())==11110 and int(o.status.eq('cancelled').sum())==1199
assert int(o.loc[o.status.eq('completed'),'delivery_minutes'].isna().sum())==0
assert int(o.loc[o.status.eq('cancelled'),'delivery_minutes'].isna().sum())==1199
assert round(o.loc[o.status.eq('completed'),'order_amount'].sum(),2)==2998668.80
with sqlite3.connect(R/'data/instamart.db') as con:
 assert set(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'",con).name)>={'customers','products','orders'}
print('ALL CHECKPOINTS PASSED')
