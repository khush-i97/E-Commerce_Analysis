```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS 
from datetime import datetime
import plotly
plotly.offline.init_notebook_mode (connected = True)
import plotly.express as px
import matplotlib.pyplot as plt
import squarify
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA,PCA,TruncatedSVD
import spacy
import en_core_web_sm
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import calendar
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.32.0.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




```python
def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va                       # Vertically align label
        )

```


```python
import os

file_path = "C:/Users/Khushi Gangrade/Desktop/projectss/E-commerce Analysis/data1.csv"
print("Exists:", os.path.exists(file_path))
```

    Exists: True



```python
df = pd.read_csv(
    r"C:\Users\Khushi Gangrade\Desktop\projectss\E-commerce Analysis\data1.csv",
    parse_dates=['InvoiceDate'],
    encoding='ISO-8859-1'  # or 'latin1'
)
df_original = df.copy()
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.55</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.75</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>3</th>
      <td>536365</td>
      <td>84029G</td>
      <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>4</th>
      <td>536365</td>
      <td>84029E</td>
      <td>RED WOOLLY HOTTIE WHITE HEART.</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>541909.000000</td>
      <td>541909</td>
      <td>541909.000000</td>
      <td>406829.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9.552250</td>
      <td>2011-07-04 13:34:57.156386048</td>
      <td>4.611114</td>
      <td>15287.690570</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-80995.000000</td>
      <td>2010-12-01 08:26:00</td>
      <td>-11062.060000</td>
      <td>12346.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>2011-03-28 11:34:00</td>
      <td>1.250000</td>
      <td>13953.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2011-07-19 17:17:00</td>
      <td>2.080000</td>
      <td>15152.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10.000000</td>
      <td>2011-10-19 11:27:00</td>
      <td>4.130000</td>
      <td>16791.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80995.000000</td>
      <td>2011-12-09 12:50:00</td>
      <td>38970.000000</td>
      <td>18287.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>218.081158</td>
      <td>NaN</td>
      <td>96.759853</td>
      <td>1713.600303</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['Quantity', 'UnitPrice', 'CustomerID']].isnull().sum()

```




    Quantity           0
    UnitPrice          0
    CustomerID    134658
    dtype: int64




```python
sns.pairplot(df[['Quantity', 'UnitPrice', 'CustomerID']].sample(1000, random_state=42))

```




    <seaborn.axisgrid.PairGrid at 0x1f12c94f140>




    
![png](output_7_1.png)
    



```python
fig,ax = plt.subplots(nrows=2,figsize=(20,7))
sns.boxplot(df[(df['InvoiceNo'].str[0]=='c')|(df['InvoiceNo'].str[0]=='C')]['Quantity'],ax=ax[0])
sns.boxplot(df[(df['InvoiceNo'].str[0]=='c')|(df['InvoiceNo'].str[0]=='C')]['UnitPrice'],ax=ax[1])
ax[0].title.set_text("Cancelled Orders 'C' quantity and price distribution")
```


    
![png](output_8_0.png)
    



```python
neg_qty = df[df["Quantity"]<0]
neg_qty_without_C = neg_qty[neg_qty["InvoiceNo"].str[0]!="C"]
print("Negative Qty without 'C' in InvoiceNo \n Unit Prices: {} \t CustomerIDs: {}".format(neg_qty_without_C["UnitPrice"].unique(),neg_qty_without_C["CustomerID"].unique()))
```

    Negative Qty without 'C' in InvoiceNo 
     Unit Prices: [0.] 	 CustomerIDs: [nan]



```python
def check_hypothesis_cancelled_order(df):
    failed = 0
    passed = 0
    neg_qty = df[df["Quantity"]<0]
    pos_qty = df[~df["Quantity"]<0]
    for ind in neg_qty.index:
        if(neg_qty['CustomerID'][ind]):
            p = pos_qty[
                (pos_qty['CustomerID'] == neg_qty['CustomerID'][ind])&
                (pos_qty['Quantity'] <= abs(neg_qty['Quantity'][ind]))&
                ((pos_qty['InvoiceDate'] - neg_qty['InvoiceDate'][ind]).dt.total_seconds()>=0)
            ]
            if(len(p)==0):
                failed+=1
            else:
                passed+=1
    if(failed>passed):
        print("Hypothesis Rejected!")
        print("Failed Counts:"+str(failed)+" Passed Counts:"+str(passed))
        print("Approximately "+str(int(failed/(failed + passed)*100)) + "% rows didn't satisfy the condition")
    else:
        print("Hypothesis Accepted")
        print("Failed Counts:"+str(failed)+" Passed Counts:"+str(passed))
        print("Approximately "+str(int(passed/(failed + passed)*100)) + "% rows satisfy the condition")
```


```python
check_hypothesis_cancelled_order(df)
```

    Hypothesis Accepted
    Failed Counts:4665 Passed Counts:5959
    Approximately 56% rows satisfy the condition



```python
neg_price = df[df["UnitPrice"]<0]
neg_price
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>299983</th>
      <td>A563186</td>
      <td>B</td>
      <td>Adjust bad debt</td>
      <td>1</td>
      <td>2011-08-12 14:51:00</td>
      <td>-11062.06</td>
      <td>NaN</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>299984</th>
      <td>A563187</td>
      <td>B</td>
      <td>Adjust bad debt</td>
      <td>1</td>
      <td>2011-08-12 14:52:00</td>
      <td>-11062.06</td>
      <td>NaN</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop_duplicates(inplace=True)
```


```python
fig,ax = plt.subplots(figsize=(14,2))
((df.isnull().sum() / len(df))*100).plot.bar(ax=ax)
add_value_labels(ax)
ax.set_title('% of Null Values')
plt.show()
```


    
![png](output_14_0.png)
    



```python
x = pd.DataFrame(df.groupby("StockCode")["Description"].value_counts())
y = x.droplevel(level=1).index
y = y[y.duplicated()]
test = df[["StockCode","Description"]]
test = test.drop_duplicates()
test1 = test[test["StockCode"].isin(y)]
test2 = pd.DataFrame(test1.groupby("StockCode")["Description"].value_counts())
test2.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>StockCode</th>
      <th>Description</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">10080</th>
      <th>GROOVY CACTUS INFLATABLE</th>
      <td>1</td>
    </tr>
    <tr>
      <th>check</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">10133</th>
      <th>COLOURING PENCILS BROWN TUBE</th>
      <td>1</td>
    </tr>
    <tr>
      <th>damaged</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">15058A</th>
      <th>BLUE POLKADOT GARDEN PARASOL</th>
      <td>1</td>
    </tr>
    <tr>
      <th>wet/rusty</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">15058C</th>
      <th>ICE CREAM DESIGN GARDEN PARASOL</th>
      <td>1</td>
    </tr>
    <tr>
      <th>wet/rusty</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">16008</th>
      <th>SMALL FOLDING SCISSOR(POINTED EDGE)</th>
      <td>1</td>
    </tr>
    <tr>
      <th>check</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(15,4))
grouped = df.groupby("StockCode")['Description'].unique()
grouped_counts = grouped.apply(lambda x: len(x)).sort_values(ascending=False)
grouped_counts.head(50).plot.bar(ax=ax)
```




    <Axes: xlabel='StockCode'>




    
![png](output_16_1.png)
    



```python
df[df["StockCode"]=="20713"]["Description"].unique()
```




    array(['JUMBO BAG OWLS', nan, 'wrongly marked. 23343 in box',
           'wrongly coded-23343', 'found', 'Found', 'wrongly marked 23343',
           'Marked as 23343', 'wrongly coded 23343'], dtype=object)




```python
def get_product_name(x):
    max_upper_count = 0
    product_name = ''
    for i in x:
        if(i==i):  #To Check for NaN
            count = 0
            for letter in i:
                if(letter.isupper()):
                    count = count+1
            if count>max_upper_count:
                max_upper_count = count
                product_name = i
    return product_name
```


```python
grouped = df.groupby("StockCode")['Description'].unique()
lookup = grouped.apply(get_product_name)
# lookup.to_excel('lookup_product_stockCode.xlsx')
```


```python
df = df.join(other=lookup, on='StockCode', how='left', rsuffix='ProductName')
df = df.rename(columns={'DescriptionProductName':'ProductName'})
```


```python
# GETTING SIMILARITY BETWEEN THE Description AND ProductName
# !pip install jellyfish
# import jellyfish
from difflib import SequenceMatcher

des = df['Description']
prod = df['ProductName']
dist = []
for d,p in zip(des, prod):
    try:
        dist.append(SequenceMatcher(None,d,p).ratio())
#         dist.append(float(jellyfish.damerau_levenshtein_distance(d,p)))
    except:
        dist.append(0)
```


```python
df['dist'] = dist
df[(df['dist']<0.3)&(df['dist']!=0)][['StockCode','Description','ProductName','dist']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StockCode</th>
      <th>Description</th>
      <th>ProductName</th>
      <th>dist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22296</th>
      <td>46000S</td>
      <td>Dotcom sales</td>
      <td>POLYESTER FILLER PAD 40x40cm</td>
      <td>0.150000</td>
    </tr>
    <tr>
      <th>22297</th>
      <td>46000M</td>
      <td>Dotcom sales</td>
      <td>POLYESTER FILLER PAD 45x45cm</td>
      <td>0.150000</td>
    </tr>
    <tr>
      <th>30555</th>
      <td>22734</td>
      <td>amazon sales</td>
      <td>SET OF 6 RIBBONS VINTAGE CHRISTMAS</td>
      <td>0.043478</td>
    </tr>
    <tr>
      <th>39047</th>
      <td>85135B</td>
      <td>Found</td>
      <td>BLUE DRAGONFLY HELICOPTER</td>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>42564</th>
      <td>22501</td>
      <td>reverse 21/5/10 adjustment</td>
      <td>PICNIC BASKET WICKER LARGE</td>
      <td>0.076923</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>524369</th>
      <td>23406</td>
      <td>CHECK</td>
      <td>HOME SWEET HOME KEY HOLDER</td>
      <td>0.193548</td>
    </tr>
    <tr>
      <th>524622</th>
      <td>22927</td>
      <td>Amazon</td>
      <td>GREEN GIANT GARDEN THERMOMETER</td>
      <td>0.055556</td>
    </tr>
    <tr>
      <th>532724</th>
      <td>22481</td>
      <td>wet boxes</td>
      <td>BLACK TEA TOWEL CLASSIC DESIGN</td>
      <td>0.051282</td>
    </tr>
    <tr>
      <th>535329</th>
      <td>21693</td>
      <td>mixed up</td>
      <td>SMALL HAMMERED SILVER CANDLEPOT</td>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>535330</th>
      <td>21688</td>
      <td>mixed up</td>
      <td>SILVER PLATE CANDLE BOWL SMALL</td>
      <td>0.052632</td>
    </tr>
  </tbody>
</table>
<p>196 rows × 4 columns</p>
</div>




```python
df["TotalPrice"] = df["UnitPrice"] * df["Quantity"]
```


```python
fig, ax = plt.subplots(figsize=(7,10))
neg_qty = df[df["Quantity"]<0]
neg_qty["TotalPrice"] = abs(neg_qty["TotalPrice"])
x = neg_qty[["ProductName","TotalPrice"]]
x.groupby("ProductName")["TotalPrice"].sum().sort_values(ascending=True).tail(30).plot.barh(ax=ax)
```




    <Axes: ylabel='ProductName'>




    
![png](output_24_1.png)
    



```python
cancelled_df = df[df['InvoiceNo'].str[0]=='C']
df = df[~(df['InvoiceNo'].str[0]=='C')]
cancelled_df = cancelled_df.reset_index(drop=True)
```


```python
fig, ax = plt.subplots(nrows=4, ncols=1,figsize=(15,20))
rev = df[df['TotalPrice']>=0]
rev['TransactionsCount'] = 1
rev = rev.groupby(rev['InvoiceDate'].dt.date).agg({'TotalPrice':'sum',
                                                  'Quantity': 'sum',
                                                  'CustomerID': 'count',
                                                  'TransactionsCount':'sum'})
rev['10 Days Moving Average Revenue'] = rev['TotalPrice'].rolling(10).mean()
rev['10 Days Moving Average Quantity'] = rev['Quantity'].rolling(10).mean()
rev['10 Days Moving Transactions Count'] = rev['TransactionsCount'].rolling(10).mean()
cust = df.groupby('CustomerID').first().reset_index()[['CustomerID','InvoiceDate']]
cust = cust.groupby(cust.InvoiceDate.dt.date).agg({'CustomerID':'count'})
cust['10 Days Moving Average Quantity'] = cust['CustomerID'].rolling(10).mean()

sns.set_style("whitegrid")
sns.lineplot(data=rev[['TotalPrice','10 Days Moving Average Revenue']], palette='magma_r', linewidth=1.5, ax=ax[0],legend=False)
ax[0].legend(title='Revenue Trends', loc='upper left', labels=['Revenue', '10 Days Moving Average Revenue'])
ax[0].title.set_text('Revenue Trends')
ax[0].set_xlabel('')

sns.lineplot(data=rev[['TotalPrice','10 Days Moving Average Quantity']], palette='ocean', linewidth=1.5, ax=ax[1])
ax[1].legend(title='Quantity Trends', loc='upper left', labels=['Quantity Sold', '10 Days Moving Average Quantity'])
ax[1].title.set_text('Quantity Sold Trends')
ax[1].set_xlabel('')

sns.lineplot(data=cust, palette='cividis', linewidth=1.5, ax=ax[2])
ax[2].legend(title='New Customers Trends', loc='upper right', labels=['New Customers', '10 Days Moving Average New Customers'])
ax[2].title.set_text('New Customers Trends')
ax[2].set_xlabel('')

sns.lineplot(data=rev[['TransactionsCount','10 Days Moving Transactions Count']], palette='twilight_shifted', linewidth=1.5, ax=ax[3])
ax[3].legend(title='Transactions Count Trend', loc='upper right', labels=['Transactions Count', '10 Days Moving Average Transactions Count'])
ax[3].title.set_text('Transactions Count Trends')
ax[3].set_xlabel('')

plt.show()
```


    
![png](output_26_0.png)
    



```python
sales_comp = df[(df['InvoiceDate'].dt.month==12)&(df['TotalPrice']>=0)][['InvoiceDate','TotalPrice','Quantity']]
sales_comp['Transactions Count'] = 1
sales_comp = sales_comp.groupby(sales_comp['InvoiceDate'].dt.year)[['TotalPrice','Quantity','Transactions Count']].sum()
fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(20,5))

sns.set_style("whitegrid")
sns.barplot(data=sales_comp, x=sales_comp.index, y='TotalPrice', palette='magma_r', ax=ax[0])
ax[0].title.set_text('Revenue Comparision')
ax[0].set_ylabel('Revenue')
ax[0].set_xlabel('December of Year')
add_value_labels(ax[0])

sns.barplot(data=sales_comp, x=sales_comp.index, y='Quantity',  palette='ocean', ax=ax[1])
ax[1].title.set_text('Quantity Sold Comparision')
add_value_labels(ax[1])
ax[1].set_xlabel('December of Year')

sns.barplot(data=sales_comp, x=sales_comp.index, y='Transactions Count',  palette='twilight_shifted', ax=ax[2])
ax[2].title.set_text('Transactions Count Comparision')
add_value_labels(ax[2])
ax[2].set_xlabel('December of Year')

fig.suptitle('Comparision for the month of December in 2020 and 2021',fontsize=16)

plt.show()
```


    
![png](output_27_0.png)
    



```python
print("Sales Revenue Difference: {:2.2f}% decline in revenue from 2010 \nSales Quantity Difference: {:2.2f}% decline in quantity from 2010".format(
((sales_comp['TotalPrice'][2010] - sales_comp['TotalPrice'][2011]) / sales_comp['TotalPrice'][2010])*100,
    ((sales_comp['Quantity'][2010] - sales_comp['Quantity'][2011]) / sales_comp['Quantity'][2010])*100
))
```

    Sales Revenue Difference: 22.36% decline in revenue from 2010 
    Sales Quantity Difference: 13.38% decline in quantity from 2010



```python

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,5))
sns.set_style("whitegrid")

week = df[df['TotalPrice']>=0][['InvoiceDate','TotalPrice','Quantity']]
week = week.groupby(week['InvoiceDate'].dt.weekday)[['TotalPrice','Quantity']].sum()
week = week.reset_index()
week['Week'] = week['InvoiceDate'].apply(lambda x: calendar.day_name[x])

sns.lineplot(data = week, x=week.Week, y='Quantity', marker='o', sort = False, ax=ax)
ax2 = ax.twinx()
sns.barplot(data = week, x=week.Week, y='TotalPrice', alpha=0.5, ax=ax2)
fig.suptitle('Revenue and Quantity by Sale Week Day Wise',fontsize=16)
add_value_labels(ax2)

plt.show()
```


    
![png](output_29_0.png)
    



```python
fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(15,7))
sns.set_style("whitegrid")

day = df[df['TotalPrice']>=0][['InvoiceDate','TotalPrice','Quantity']]
day = day.groupby(day['InvoiceDate'].dt.hour)[['TotalPrice','Quantity']].sum()

sns.barplot(data = day, x=day.index, y='TotalPrice', alpha=1, ax=ax[0])
sns.lineplot(data = day, x=day.index, y='Quantity', marker='o', sort = False, ax=ax[1])
fig.suptitle('Revenue and Quantity by Sale Hourwise',fontsize=16)
add_value_labels(ax[0])
plt.show()
```


    
![png](output_30_0.png)
    



```python
fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(15,7))
sns.set_style("whitegrid")

date = df[df['TotalPrice']>=0][['InvoiceDate','TotalPrice','Quantity']]
date = date.groupby(date['InvoiceDate'].dt.day)[['TotalPrice','Quantity']].sum()

sns.barplot(data = date, x=date.index, y='TotalPrice', alpha=1, ax=ax[0])
sns.lineplot(data = date, x=date.index, y='Quantity', marker='o', sort = False, ax=ax[1])
fig.suptitle('Revenue and Quantity by Sale Daywise',fontsize=16)

plt.show()
```


    
![png](output_31_0.png)
    



```python
fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(15,7))
sns.set_style("whitegrid")

q = df[(df['TotalPrice']>=0)&(df['InvoiceDate'].dt.year==2011)][['InvoiceDate','TotalPrice','Quantity']]
q = q.groupby(q['InvoiceDate'].dt.quarter)[['TotalPrice','Quantity']].sum()

sns.barplot(data = q, x=q.index, y='TotalPrice', alpha=0.7, ax=ax[0])
sns.lineplot(data = q, x=q.index, y='Quantity', marker='o', sort = False, ax=ax[1])
fig.suptitle('Revenue and Quantity by Sale Quarterly for 2011',fontsize=16)
add_value_labels(ax[0])
ax[1].set_xticklabels(['',1,'',2,'',3,'',4])
plt.show()
```


    
![png](output_32_0.png)
    



```python
reg = df[df['TotalPrice']>=0].groupby('Country').agg({'TotalPrice':'sum',
                                                  'Quantity': 'sum',
                                                  'CustomerID': 'count'})
```


```python
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,30))
g1 = sns.barplot(x=reg['TotalPrice'], y=reg.index, alpha=1, ax=ax[0],palette='Reds', orient='h')
g2 = sns.barplot(x=reg['Quantity'], y=reg.index, alpha=1, ax=ax[1], palette='Blues',orient='h')
g3 = sns.barplot(x=reg['CustomerID'], y=reg.index, alpha=1, ax=ax[2], palette='Greens', orient='h')
ax[2].title.set_text('Customers Count by Country')
ax[2].set_xlabel("Customers (Log Scale)")
ax[1].title.set_text('Quantity Sold by Country')
ax[1].set_xlabel("Quantity (Log Scale)")
ax[0].title.set_text('Revenue by Country')
ax[0].set_xlabel("Revenue (Log Scale)")
g1.set_xscale("log")
g2.set_xscale("log")
g3.set_xscale("log")
plt.show()
```


    
![png](output_34_0.png)
    



```python
reg = reg[reg.index!='United Kingdom']
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20,20))
# Change color
squarify.plot(sizes=reg['TotalPrice'], label=[str(x)+'\n'+str(y)+'K' for x,y in zip(reg.index,(reg['TotalPrice']/1000).round(2))], alpha=.6, ax=ax[0], color=mcolors.CSS4_COLORS )
ax[0].title.set_text('Revenue by Country (Excluding UK)')
squarify.plot(sizes=reg['Quantity'], label=[str(x)+'\n'+str(y)+'K' for x,y in zip(reg.index,(reg['Quantity']/1000).round(2))], alpha=.6, ax=ax[1], color=mcolors.CSS4_COLORS )
ax[1].title.set_text('Quantity Sold by Country (Excluding UK)')
r1 = reg[reg['CustomerID']!=0]
squarify.plot(sizes=r1['CustomerID'], label=[str(x)+'\n'+str(y)+'K' for x,y in zip(r1.index,(r1['CustomerID']/1000).round(2))], alpha=.6, ax=ax[2], color=mcolors.CSS4_COLORS )
ax[2].title.set_text('Customers Count by Country (Excluding UK)')
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
plt.show()
```


    
![png](output_35_0.png)
    



```python
## https://www.kaggle.com/fabiendaniel/customer-segmentation


import plotly.graph_objs as go
import warnings
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")

temp = df[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
temp = temp.reset_index(drop = False)
countries = temp['Country'].value_counts()

data = dict(type='choropleth',
locations = countries.index,
locationmode = 'country names', z = countries,
text = countries.index, colorbar = {'title':'Order no.'},
colorscale=[[0, 'rgb(224,255,255)'],
            [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],
            [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],
            [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],
            [1, 'rgb(227,26,28)']],    
reversescale = False)
#_______________________
layout = dict(title='Number of orders per country',
geo = dict(showframe = True, projection={'type':'mercator'}))
#______________
choromap = go.Figure(data = [data], layout = layout)
iplot(choromap, validate=False)
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.32.0.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




<div>                            <div id="181c963b-ebaa-4c0e-be63-f9bdc7c37124" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("181c963b-ebaa-4c0e-be63-f9bdc7c37124")) {                    Plotly.newPlot(                        "181c963b-ebaa-4c0e-be63-f9bdc7c37124",                        [{"colorbar":{"title":{"text":"Order no."}},"colorscale":[[0,"rgb(224,255,255)"],[0.01,"rgb(166,206,227)"],[0.02,"rgb(31,120,180)"],[0.03,"rgb(178,223,138)"],[0.05,"rgb(51,160,44)"],[0.1,"rgb(251,154,153)"],[0.2,"rgb(255,255,0)"],[1,"rgb(227,26,28)"]],"locationmode":"country names","locations":["United Kingdom","Germany","France","EIRE","Belgium","Netherlands","Spain","Portugal","Australia","Switzerland","Finland","Italy","Norway","Sweden","Channel Islands","Japan","Poland","Denmark","Austria","Cyprus","Unspecified","Iceland","Singapore","Canada","Greece","USA","Israel","Malta","European Community","Lithuania","United Arab Emirates","Czech Republic","Bahrain","Saudi Arabia","Lebanon","Brazil","RSA"],"reversescale":false,"text":["United Kingdom","Germany","France","EIRE","Belgium","Netherlands","Spain","Portugal","Australia","Switzerland","Finland","Italy","Norway","Sweden","Channel Islands","Japan","Poland","Denmark","Austria","Cyprus","Unspecified","Iceland","Singapore","Canada","Greece","USA","Israel","Malta","European Community","Lithuania","United Arab Emirates","Czech Republic","Bahrain","Saudi Arabia","Lebanon","Brazil","RSA"],"z":[16649,457,389,260,98,95,90,57,57,51,41,38,36,36,26,19,19,18,17,16,8,7,7,6,5,5,5,5,4,4,3,2,2,1,1,1,1],"type":"choropleth"}],                        {"geo":{"projection":{"type":"mercator"},"showframe":true},"title":{"text":"Number of orders per country"},"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('181c963b-ebaa-4c0e-be63-f9bdc7c37124');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
lastdate = datetime(2012,1,1)
cleaned_dataset = df[df['TotalPrice']>=0]
recent = (lastdate - cleaned_dataset.groupby("CustomerID")["InvoiceDate"].last()).dt.days
frequent = cleaned_dataset.groupby("CustomerID")["InvoiceDate"].count()
monetary = cleaned_dataset.groupby("CustomerID")["TotalPrice"].sum()
```


```python
recent_quantile = recent.quantile(q=[0.25,0.5,0.75])
recent_quantile
```




    0.25     39.0
    0.50     72.0
    0.75    163.5
    Name: InvoiceDate, dtype: float64




```python
frequent_quantile = frequent.quantile(q=[0.25,0.5,0.75])
frequent_quantile
```




    0.25    17.0
    0.50    41.0
    0.75    98.0
    Name: InvoiceDate, dtype: float64




```python
monetary_quantile = monetary.quantile(q=[0.25,0.5,0.75])
monetary_quantile
```




    0.25     306.455
    0.50     668.560
    0.75    1660.315
    Name: TotalPrice, dtype: float64




```python
rfm = pd.DataFrame(data=[recent,frequent,monetary])
rfm = rfm.transpose()
rfm.columns = ["recent","frequent","monetary"]
rfm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recent</th>
      <th>frequent</th>
      <th>monetary</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12346.0</th>
      <td>347.0</td>
      <td>1.0</td>
      <td>77183.60</td>
    </tr>
    <tr>
      <th>12347.0</th>
      <td>24.0</td>
      <td>182.0</td>
      <td>4310.00</td>
    </tr>
    <tr>
      <th>12348.0</th>
      <td>97.0</td>
      <td>31.0</td>
      <td>1797.24</td>
    </tr>
    <tr>
      <th>12349.0</th>
      <td>40.0</td>
      <td>73.0</td>
      <td>1757.55</td>
    </tr>
    <tr>
      <th>12350.0</th>
      <td>332.0</td>
      <td>17.0</td>
      <td>334.40</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18280.0</th>
      <td>299.0</td>
      <td>10.0</td>
      <td>180.60</td>
    </tr>
    <tr>
      <th>18281.0</th>
      <td>202.0</td>
      <td>7.0</td>
      <td>80.82</td>
    </tr>
    <tr>
      <th>18282.0</th>
      <td>29.0</td>
      <td>12.0</td>
      <td>178.05</td>
    </tr>
    <tr>
      <th>18283.0</th>
      <td>25.0</td>
      <td>721.0</td>
      <td>2045.53</td>
    </tr>
    <tr>
      <th>18287.0</th>
      <td>64.0</td>
      <td>70.0</td>
      <td>1837.28</td>
    </tr>
  </tbody>
</table>
<p>4339 rows × 3 columns</p>
</div>




```python
def get_kmeans_wcss(data, n_limit=15):
    wcss = [] #Within cluster sum of squares (WCSS)
    for i in range(1,n_limit):
        km = KMeans(init='k-means++', n_clusters=i, n_init=10)
        km.fit(data)
        wcss.append(km.inertia_)
    plt.title("Elbow Method")
    plt.plot(range(1, n_limit), wcss)
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    return wcss
```


```python
_ = get_kmeans_wcss(rfm, n_limit=15)
```


    
![png](output_43_0.png)
    



```python
kmeans = KMeans(n_clusters=3, init = "k-means++", random_state=42)
clustered_cust = kmeans.fit_predict(rfm)
```


```python
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))

# Boxplot for Recency
sns.boxplot(x=clustered_cust, y=recent, palette="cubehelix", ax=ax[0])
ax[0].set(xlabel="Clusters", ylabel="Recency in Number of Days")
ax[0].set_title('Clusters on Recency')

# Boxplot for Frequency
sns.boxplot(x=clustered_cust, y=frequent, palette="cubehelix", ax=ax[1])
ax[1].set(xlabel="Clusters", ylabel="Frequency in Number of Days")
ax[1].set_title('Clusters on Frequency')

# Boxplot for Monetary
sns.boxplot(x=clustered_cust, y=monetary, palette="cubehelix", ax=ax[2])
ax[2].set(xlabel="Clusters", ylabel="Spending Amount")
ax[2].set_title('Clusters on Monetary')

plt.tight_layout()
plt.show()

```


    
![png](output_45_0.png)
    



```python
rfm['Clusters'] = clustered_cust
rfm.Clusters.value_counts()
```




    Clusters
    0    4301
    2      32
    1       6
    Name: count, dtype: int64




```python
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Combine all product descriptions into one big string efficiently
comment_words = ' '.join(
    df['Description']
    .dropna()  # Drop missing descriptions
    .astype(str)
    .str.lower()  # Lowercase
)

# Define stopwords
stopwords = set(STOPWORDS)

# Generate WordCloud
wordcloud = WordCloud(
    width=1200, height=600,
    background_color='white',
    stopwords=stopwords,
    min_font_size=10
).generate(comment_words)

# Plot
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

```


    
![png](output_47_0.png)
    



```python
df['HolidaySeason'] = 0
df.loc[df['InvoiceDate'].dt.month.isin([9,10,11]), 'HolidaySeason'] = 1
```


```python
d = df[df['InvoiceDate'].dt.year==2011]
d['Transactions Count'] = 1
labels0 = ['Holiday Season Revenue', 'Non-Holiday Revenue']
sizes0 = [d[d['HolidaySeason']==1]['TotalPrice'].sum(),
         d[d['HolidaySeason']==0]['TotalPrice'].sum()
        ]

labels1 = ['Holiday Season Quantity', 'Non-Holiday Quantity']
sizes1 = [d[d['HolidaySeason']==1]['Quantity'].sum(),
         d[d['HolidaySeason']==0]['Quantity'].sum()
        ]

labels2 = ['Holiday Season Transactions Count', 'Non-Holiday Transactions Count']
sizes2 = [d[d['HolidaySeason']==1]['Transactions Count'].sum(),
         d[d['HolidaySeason']==0]['Transactions Count'].sum()
        ]

fig1, ax = plt.subplots(ncols=3,figsize=(18,5))
ax[0].pie(sizes0, labels=labels0, autopct='%1.1f%%', shadow=True)
ax[0].axis('equal')
ax[1].pie(sizes1, labels=labels1, autopct='%1.1f%%', shadow=True)
ax[1].axis('equal')
ax[2].pie(sizes2, labels=labels2, autopct='%1.1f%%', shadow=True)
ax[2].axis('equal')
plt.show()
```


    
![png](output_49_0.png)
    



```python
df['Transactions Count'] = 1
l1 = df[df['HolidaySeason']==1].groupby('StockCode')['Transactions Count'].sum()
l2 = df[df['HolidaySeason']==0].groupby('StockCode')['Transactions Count'].sum()
x = pd.DataFrame(data=[l1,l2]).T
x.columns = ['Season','Off-Season']
x = x.fillna(0)
x = x.reset_index()
```


```python
_ = get_kmeans_wcss(x[['Season','Off-Season']], n_limit=20)
```


    
![png](output_51_0.png)
    



```python
kmeans = KMeans(n_clusters=10, init = "k-means++", random_state=100)
clustered_cust = kmeans.fit_predict(x[['Season','Off-Season']])
x['cluster'] = clustered_cust
```


```python
plt.figure(figsize=(15,10))

g1 = sns.scatterplot(
    data=x,
    x="Season",
    y="Off-Season",
    hue="cluster",
    palette="deep"
)

g1.set_xscale("log")
plt.xlabel('Season Counts')
plt.title('StockCodes: Seasonal and Off-seasonal counts by KNN clusters (Seasonal Count axis is in log scale)')
plt.ylabel('Off-Season Counts')
plt.show()

```


    
![png](output_53_0.png)
    



```python
# Step 1: Get StockCodes from cluster 8
stockcodes_cluster_8 = x[x['cluster'] == 8].index

# Step 2: Get matching product names from the main DataFrame
df[df['StockCode'].isin(stockcodes_cluster_8)]['Description'].unique()

```




    array([], dtype=object)




```python

QUANTILE = [0.90]
MAX_QUANTILE = [0.95]
MIN_QUANTILE = [0.15]
print(x['Season'].quantile(QUANTILE))
print(x['Off-Season'].quantile(QUANTILE))
x.loc[:,'Q-Region'] = 0
x.loc[(x['Season']>x['Season'].quantile(QUANTILE).values[0])&(x['Off-Season']>x['Off-Season'].quantile(QUANTILE).values[0]),'Q-Region'] = 1
x.loc[(x['Season']<=x['Season'].quantile(QUANTILE).values[0])&(x['Off-Season']>x['Off-Season'].quantile(QUANTILE).values[0]),'Q-Region'] = 2
x.loc[(x['Season']<=x['Season'].quantile(QUANTILE).values[0])&(x['Off-Season']<=x['Off-Season'].quantile(QUANTILE).values[0]),'Q-Region'] = 3
x.loc[(x['Season']>x['Season'].quantile(QUANTILE).values[0])&(x['Off-Season']<=x['Off-Season'].quantile(QUANTILE).values[0]),'Q-Region'] = 4
```

    0.9    132.0
    Name: Season, dtype: float64
    0.9    215.0
    Name: Off-Season, dtype: float64



```python
plt.figure(figsize=(15, 10))

g1 = sns.scatterplot(
    data=x,
    x='Season',
    y='Off-Season',
    hue='Q-Region',
    palette="deep"
)

# Uncomment if you want log scale
# g1.set_xscale("log")

plt.title('StockCodes: Seasonal and Off-seasonal counts by Quantile Regions (Seasonal Count axis is in log scale)', fontsize=16)
plt.xlabel('Season Counts', fontsize=13)
plt.ylabel('Off-Season Counts', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Q-Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

```


    
![png](output_56_0.png)
    



```python
x.head()
x.index.name
x.columns

```




    Index(['StockCode', 'Season', 'Off-Season', 'cluster', 'Q-Region'], dtype='object')




```python
# Step 1: Make StockCode a column
x_reset = x.reset_index()  # Now StockCode is a column again

# Step 2: Merge on 'StockCode'
df_merged = df.merge(
    x_reset,
    on='StockCode',  # now both sides have this column
    how='left'
)

# Step 3: Drop columns and rename
df_merged = df_merged.drop(['Season', 'Off-Season'], axis=1)
df_merged = df_merged.rename(columns={'cluster': 'ProductCluster'})

```


```python
df = df_merged

```


```python
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 14))

# Filter data to valid quantities and year 2011
ddf = df[(df['Quantity'] > 0) & (df['InvoiceDate'].dt.year == 2011)]

# Group by week number and Q-Region
d = ddf.groupby([
    ddf['InvoiceDate'].dt.isocalendar().week,
    ddf['Q-Region']
]).agg({
    'TotalPrice': 'sum',
    'Quantity': 'sum',
    'CustomerID': 'count'
}).reset_index().rename(columns={'week': 'Week'})

# Plot total price over time
sns.lineplot(data=d, x='Week', y='TotalPrice', hue='Q-Region', palette='deep', ax=ax[0])
ax[0].set_title('Total Sales by Week and Q-Region', fontsize=14)

# Plot quantity over time
sns.lineplot(data=d, x='Week', y='Quantity', hue='Q-Region', palette='deep', ax=ax[1])
ax[1].set_title('Quantity Sold by Week and Q-Region', fontsize=14)

plt.tight_layout()
plt.show()

```


    
![png](output_60_0.png)
    



```python
Q1 = df[df['Q-Region']==1]
Q2 = df[df['Q-Region']==2]
Q3 = df[df['Q-Region']==3]
Q4 = df[df['Q-Region']==4]
```


```python
Q1.groupby(['StockCode','ProductName'])[['UnitPrice','TotalPrice','Quantity']].sum().sort_values(by='UnitPrice',ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>UnitPrice</th>
      <th>TotalPrice</th>
      <th>Quantity</th>
    </tr>
    <tr>
      <th>StockCode</th>
      <th>ProductName</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DOT</th>
      <th>DOTCOM POSTAGE</th>
      <td>206248.77</td>
      <td>206248.77</td>
      <td>1708</td>
    </tr>
    <tr>
      <th>POST</th>
      <th>POSTAGE</th>
      <td>34992.23</td>
      <td>78101.88</td>
      <td>6500</td>
    </tr>
    <tr>
      <th>22423</th>
      <th>REGENCY CAKESTAND 3 TIER</th>
      <td>28065.76</td>
      <td>174156.54</td>
      <td>13809</td>
    </tr>
    <tr>
      <th>47566</th>
      <th>PARTY BUNTING</th>
      <td>9850.68</td>
      <td>99445.23</td>
      <td>18287</td>
    </tr>
    <tr>
      <th>22720</th>
      <th>SET OF 3 CAKE TINS PANTRY DESIGN</th>
      <td>8120.53</td>
      <td>38108.89</td>
      <td>7433</td>
    </tr>
    <tr>
      <th>85066</th>
      <th>CREAM SWEETHEART MINI CHEST</th>
      <td>7497.46</td>
      <td>22594.20</td>
      <td>1784</td>
    </tr>
    <tr>
      <th>85123A</th>
      <th>WHITE HANGING HEART T-LIGHT HOLDER</th>
      <td>7024.49</td>
      <td>104462.75</td>
      <td>41389</td>
    </tr>
    <tr>
      <th>22624</th>
      <th>IVORY KITCHEN SCALES</th>
      <td>6727.86</td>
      <td>16378.71</td>
      <td>1910</td>
    </tr>
    <tr>
      <th>22847</th>
      <th>BREAD BIN DINER STYLE IVORY</th>
      <td>6545.63</td>
      <td>14389.34</td>
      <td>896</td>
    </tr>
    <tr>
      <th>23284</th>
      <th>DOORMAT KEEP CALM AND COME IN</th>
      <td>6424.24</td>
      <td>38133.64</td>
      <td>5485</td>
    </tr>
  </tbody>
</table>
</div>




```python
Q2.groupby(['StockCode','ProductName'])[['UnitPrice','TotalPrice','Quantity']].sum().sort_values(by='UnitPrice',ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>UnitPrice</th>
      <th>TotalPrice</th>
      <th>Quantity</th>
    </tr>
    <tr>
      <th>StockCode</th>
      <th>ProductName</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22424</th>
      <th>ENAMEL BREAD BIN CREAM</th>
      <td>7216.03</td>
      <td>12358.33</td>
      <td>873</td>
    </tr>
    <tr>
      <th>21843</th>
      <th>RED RETROSPOT CAKE STAND</th>
      <td>6820.69</td>
      <td>21354.30</td>
      <td>1949</td>
    </tr>
    <tr>
      <th>22501</th>
      <th>PICNIC BASKET WICKER LARGE</th>
      <td>6408.71</td>
      <td>19302.26</td>
      <td>1565</td>
    </tr>
    <tr>
      <th>22605</th>
      <th>WOODEN CROQUET GARDEN SET</th>
      <td>4867.47</td>
      <td>11276.21</td>
      <td>782</td>
    </tr>
    <tr>
      <th>22502</th>
      <th>PICNIC BASKET WICKER 60 PIECES</th>
      <td>4800.89</td>
      <td>51408.77</td>
      <td>1945</td>
    </tr>
    <tr>
      <th>22487</th>
      <th>WHITE WOOD GARDEN PLANT LADDER</th>
      <td>4663.66</td>
      <td>7909.28</td>
      <td>784</td>
    </tr>
    <tr>
      <th>21621</th>
      <th>VINTAGE UNION JACK BUNTING</th>
      <td>4581.32</td>
      <td>24525.83</td>
      <td>2357</td>
    </tr>
    <tr>
      <th>22844</th>
      <th>VINTAGE CREAM DOG FOOD CONTAINER</th>
      <td>4526.56</td>
      <td>9661.32</td>
      <td>1003</td>
    </tr>
    <tr>
      <th>21524</th>
      <th>DOORMAT SPOTTY HOME SWEET HOME</th>
      <td>4427.39</td>
      <td>13572.06</td>
      <td>1752</td>
    </tr>
    <tr>
      <th>21217</th>
      <th>RED RETROSPOT ROUND CAKE TINS</th>
      <td>4326.04</td>
      <td>11522.44</td>
      <td>1088</td>
    </tr>
  </tbody>
</table>
</div>




```python
Q3.groupby(['StockCode','ProductName'])[['UnitPrice','TotalPrice','Quantity']].sum().sort_values(by='UnitPrice',ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>UnitPrice</th>
      <th>TotalPrice</th>
      <th>Quantity</th>
    </tr>
    <tr>
      <th>StockCode</th>
      <th>ProductName</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M</th>
      <th>Manual</th>
      <td>74098.73</td>
      <td>77750.27</td>
      <td>6990</td>
    </tr>
    <tr>
      <th>AMAZONFEE</th>
      <th>AMAZON FEE</th>
      <td>13761.09</td>
      <td>13761.09</td>
      <td>2</td>
    </tr>
    <tr>
      <th>C2</th>
      <th>CARRIAGE</th>
      <td>7033.00</td>
      <td>7051.00</td>
      <td>292</td>
    </tr>
    <tr>
      <th>84078A</th>
      <th>SET/4 WHITE RETRO STORAGE CUBES</th>
      <td>7011.00</td>
      <td>16363.30</td>
      <td>415</td>
    </tr>
    <tr>
      <th>22655</th>
      <th>VINTAGE RED KITCHEN CABINET</th>
      <td>5450.00</td>
      <td>8125.00</td>
      <td>49</td>
    </tr>
    <tr>
      <th>22846</th>
      <th>BREAD BIN DINER STYLE RED</th>
      <td>4920.45</td>
      <td>10676.62</td>
      <td>665</td>
    </tr>
    <tr>
      <th>22826</th>
      <th>LOVE SEAT ANTIQUE WHITE METAL</th>
      <td>4675.00</td>
      <td>6210.00</td>
      <td>58</td>
    </tr>
    <tr>
      <th>22827</th>
      <th>RUSTIC  SEVENTEEN DRAWER SIDEBOARD</th>
      <td>4110.00</td>
      <td>5415.00</td>
      <td>31</td>
    </tr>
    <tr>
      <th>21340</th>
      <th>CLASSIC METAL BIRDCAGE PLANT HOLDER</th>
      <td>3894.65</td>
      <td>11300.02</td>
      <td>975</td>
    </tr>
    <tr>
      <th>22839</th>
      <th>3 TIER CAKE TIN GREEN AND CREAM</th>
      <td>3757.02</td>
      <td>6981.67</td>
      <td>445</td>
    </tr>
  </tbody>
</table>
</div>




```python
Q4.groupby(['StockCode','ProductName'])[['UnitPrice','TotalPrice','Quantity']].sum().sort_values(by='UnitPrice',ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>UnitPrice</th>
      <th>TotalPrice</th>
      <th>Quantity</th>
    </tr>
    <tr>
      <th>StockCode</th>
      <th>ProductName</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23355</th>
      <th>HOT WATER BOTTLE KEEP CALM</th>
      <td>4196.16</td>
      <td>28718.80</td>
      <td>5816</td>
    </tr>
    <tr>
      <th>22114</th>
      <th>HOT WATER BOTTLE TEA AND SYMPATHY</th>
      <td>3245.81</td>
      <td>32973.44</td>
      <td>5666</td>
    </tr>
    <tr>
      <th>22941</th>
      <th>CHRISTMAS LIGHTS 10 REINDEER</th>
      <td>3098.46</td>
      <td>13795.92</td>
      <td>1670</td>
    </tr>
    <tr>
      <th>23356</th>
      <th>LOVE HOT WATER BOTTLE</th>
      <td>3015.93</td>
      <td>13334.32</td>
      <td>2359</td>
    </tr>
    <tr>
      <th>23397</th>
      <th>FOOT STOOL HOME SWEET HOME</th>
      <td>2511.53</td>
      <td>5097.65</td>
      <td>480</td>
    </tr>
    <tr>
      <th>85048</th>
      <th>15CM CHRISTMAS GLASS BALL 20 LIGHTS</th>
      <td>2447.73</td>
      <td>10910.61</td>
      <td>1359</td>
    </tr>
    <tr>
      <th>23080</th>
      <th>RED METAL BOX TOP SECRET</th>
      <td>2437.55</td>
      <td>9117.62</td>
      <td>1074</td>
    </tr>
    <tr>
      <th>23108</th>
      <th>SET OF 10 LED DOLLY LIGHTS</th>
      <td>2413.28</td>
      <td>13290.52</td>
      <td>2219</td>
    </tr>
    <tr>
      <th>23313</th>
      <th>VINTAGE CHRISTMAS BUNTING</th>
      <td>2263.84</td>
      <td>15798.66</td>
      <td>2990</td>
    </tr>
    <tr>
      <th>23328</th>
      <th>SET 6 SCHOOL MILK BOTTLES IN CRATE</th>
      <td>1981.88</td>
      <td>13269.49</td>
      <td>2841</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure
fig, ax = plt.subplots(nrows=10, ncols=1, figsize=(15, 70))

# Filter for 2011 and positive quantities
ddf = df[(df['Quantity'] > 0) & (df["InvoiceDate"].dt.year == 2011)]

# Extract week numbers
ddf['Weeks'] = ddf['InvoiceDate'].dt.isocalendar().week

# Group by week and ProductCluster
d = ddf.groupby(['Weeks', 'ProductCluster']).agg({
    'TotalPrice': 'sum',
    'Quantity': 'sum',
    'CustomerID': 'count'
}).reset_index()

# Rename for clarity
d = d.rename(columns={'TotalPrice': 'Revenue'})

# Plot for each cluster
palettes = [
    "Greens", "Reds", "deep", "tab10", "tab10_r",
    "prism", "vlag", "RdPu_r", "RdPu_r", "CMRmap"
]

for i in range(10):
    sns.lineplot(
        data=d[d['ProductCluster'] == i],
        x='Weeks',
        y='Revenue',
        hue='ProductCluster',
        palette=palettes[i],
        ax=ax[i]
    )
    ax[i].set_title(f'Product Cluster {i}', fontsize=14)

plt.tight_layout()
plt.show()

```


    
![png](output_66_0.png)
    



```python
basket_Germany = df[df['Country']=="Germany"].groupby(['InvoiceNo', 'ProductName'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
basket_EIRE = df[df['Country']=="EIRE"].groupby(['InvoiceNo', 'ProductName'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
basket_UK = df[df['Country']=="UK"].groupby(['InvoiceNo', 'ProductName'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
basket_France = df[df['Country']=="France"].groupby(['InvoiceNo', 'ProductName'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
```


```python
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_Germany.drop('POSTAGE',axis=1,inplace=True)
basket_France.drop('POSTAGE',axis=1,inplace=True)

basket_Germany = basket_Germany.applymap(encode_units)
basket_EIRE = basket_EIRE.applymap(encode_units)
basket_UK = basket_UK.applymap(encode_units)
basket_France = basket_France.applymap(encode_units)
```


```python
frequent_itemsets = apriori(basket_France, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.sort_values(['lift','support'],ascending=False).reset_index(drop=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>representativity</th>
      <th>leverage</th>
      <th>conviction</th>
      <th>zhangs_metric</th>
      <th>jaccard</th>
      <th>certainty</th>
      <th>kulczynski</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(ALARM CLOCK BAKELIKE RED )</td>
      <td>(ALARM CLOCK BAKELIKE GREEN)</td>
      <td>0.094388</td>
      <td>0.096939</td>
      <td>0.079082</td>
      <td>0.837838</td>
      <td>8.642959</td>
      <td>1.0</td>
      <td>0.069932</td>
      <td>5.568878</td>
      <td>0.976465</td>
      <td>0.704545</td>
      <td>0.820431</td>
      <td>0.826814</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(ALARM CLOCK BAKELIKE GREEN)</td>
      <td>(ALARM CLOCK BAKELIKE RED )</td>
      <td>0.096939</td>
      <td>0.094388</td>
      <td>0.079082</td>
      <td>0.815789</td>
      <td>8.642959</td>
      <td>1.0</td>
      <td>0.069932</td>
      <td>4.916181</td>
      <td>0.979224</td>
      <td>0.704545</td>
      <td>0.796590</td>
      <td>0.826814</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(ALARM CLOCK BAKELIKE PINK)</td>
      <td>(ALARM CLOCK BAKELIKE RED )</td>
      <td>0.102041</td>
      <td>0.094388</td>
      <td>0.073980</td>
      <td>0.725000</td>
      <td>7.681081</td>
      <td>1.0</td>
      <td>0.064348</td>
      <td>3.293135</td>
      <td>0.968652</td>
      <td>0.604167</td>
      <td>0.696338</td>
      <td>0.754392</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(ALARM CLOCK BAKELIKE RED )</td>
      <td>(ALARM CLOCK BAKELIKE PINK)</td>
      <td>0.094388</td>
      <td>0.102041</td>
      <td>0.073980</td>
      <td>0.783784</td>
      <td>7.681081</td>
      <td>1.0</td>
      <td>0.064348</td>
      <td>4.153061</td>
      <td>0.960466</td>
      <td>0.604167</td>
      <td>0.759214</td>
      <td>0.754392</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(SET/6 RED SPOTTY PAPER PLATES)</td>
      <td>(SET/20 RED RETROSPOT PAPER NAPKINS , SET/6 RE...</td>
      <td>0.127551</td>
      <td>0.102041</td>
      <td>0.099490</td>
      <td>0.780000</td>
      <td>7.644000</td>
      <td>1.0</td>
      <td>0.086474</td>
      <td>4.081633</td>
      <td>0.996251</td>
      <td>0.764706</td>
      <td>0.755000</td>
      <td>0.877500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(SET/20 RED RETROSPOT PAPER NAPKINS , SET/6 RE...</td>
      <td>(SET/6 RED SPOTTY PAPER PLATES)</td>
      <td>0.102041</td>
      <td>0.127551</td>
      <td>0.099490</td>
      <td>0.975000</td>
      <td>7.644000</td>
      <td>1.0</td>
      <td>0.086474</td>
      <td>34.897959</td>
      <td>0.967949</td>
      <td>0.764706</td>
      <td>0.971345</td>
      <td>0.877500</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(ALARM CLOCK BAKELIKE PINK)</td>
      <td>(ALARM CLOCK BAKELIKE GREEN)</td>
      <td>0.102041</td>
      <td>0.096939</td>
      <td>0.073980</td>
      <td>0.725000</td>
      <td>7.478947</td>
      <td>1.0</td>
      <td>0.064088</td>
      <td>3.283859</td>
      <td>0.964734</td>
      <td>0.591837</td>
      <td>0.695480</td>
      <td>0.744079</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(ALARM CLOCK BAKELIKE GREEN)</td>
      <td>(ALARM CLOCK BAKELIKE PINK)</td>
      <td>0.096939</td>
      <td>0.102041</td>
      <td>0.073980</td>
      <td>0.763158</td>
      <td>7.478947</td>
      <td>1.0</td>
      <td>0.064088</td>
      <td>3.791383</td>
      <td>0.959283</td>
      <td>0.591837</td>
      <td>0.736244</td>
      <td>0.744079</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(SET/6 RED SPOTTY PAPER PLATES, SET/20 RED RET...</td>
      <td>(SET/6 RED SPOTTY PAPER CUPS)</td>
      <td>0.102041</td>
      <td>0.137755</td>
      <td>0.099490</td>
      <td>0.975000</td>
      <td>7.077778</td>
      <td>1.0</td>
      <td>0.085433</td>
      <td>34.489796</td>
      <td>0.956294</td>
      <td>0.709091</td>
      <td>0.971006</td>
      <td>0.848611</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(SET/6 RED SPOTTY PAPER CUPS)</td>
      <td>(SET/6 RED SPOTTY PAPER PLATES, SET/20 RED RET...</td>
      <td>0.137755</td>
      <td>0.102041</td>
      <td>0.099490</td>
      <td>0.722222</td>
      <td>7.077778</td>
      <td>1.0</td>
      <td>0.085433</td>
      <td>3.232653</td>
      <td>0.995904</td>
      <td>0.709091</td>
      <td>0.690657</td>
      <td>0.848611</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(SET/6 RED SPOTTY PAPER PLATES)</td>
      <td>(SET/6 RED SPOTTY PAPER CUPS)</td>
      <td>0.127551</td>
      <td>0.137755</td>
      <td>0.122449</td>
      <td>0.960000</td>
      <td>6.968889</td>
      <td>1.0</td>
      <td>0.104878</td>
      <td>21.556122</td>
      <td>0.981725</td>
      <td>0.857143</td>
      <td>0.953609</td>
      <td>0.924444</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(SET/6 RED SPOTTY PAPER CUPS)</td>
      <td>(SET/6 RED SPOTTY PAPER PLATES)</td>
      <td>0.137755</td>
      <td>0.127551</td>
      <td>0.122449</td>
      <td>0.888889</td>
      <td>6.968889</td>
      <td>1.0</td>
      <td>0.104878</td>
      <td>7.852041</td>
      <td>0.993343</td>
      <td>0.857143</td>
      <td>0.872645</td>
      <td>0.924444</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(SET/6 RED SPOTTY PAPER PLATES, SET/6 RED SPOT...</td>
      <td>(SET/20 RED RETROSPOT PAPER NAPKINS )</td>
      <td>0.122449</td>
      <td>0.132653</td>
      <td>0.099490</td>
      <td>0.812500</td>
      <td>6.125000</td>
      <td>1.0</td>
      <td>0.083247</td>
      <td>4.625850</td>
      <td>0.953488</td>
      <td>0.639344</td>
      <td>0.783824</td>
      <td>0.781250</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(SET/20 RED RETROSPOT PAPER NAPKINS )</td>
      <td>(SET/6 RED SPOTTY PAPER PLATES, SET/6 RED SPOT...</td>
      <td>0.132653</td>
      <td>0.122449</td>
      <td>0.099490</td>
      <td>0.750000</td>
      <td>6.125000</td>
      <td>1.0</td>
      <td>0.083247</td>
      <td>3.510204</td>
      <td>0.964706</td>
      <td>0.639344</td>
      <td>0.715116</td>
      <td>0.781250</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(SET/6 RED SPOTTY PAPER PLATES)</td>
      <td>(SET/20 RED RETROSPOT PAPER NAPKINS )</td>
      <td>0.127551</td>
      <td>0.132653</td>
      <td>0.102041</td>
      <td>0.800000</td>
      <td>6.030769</td>
      <td>1.0</td>
      <td>0.085121</td>
      <td>4.336735</td>
      <td>0.956140</td>
      <td>0.645161</td>
      <td>0.769412</td>
      <td>0.784615</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(SET/20 RED RETROSPOT PAPER NAPKINS )</td>
      <td>(SET/6 RED SPOTTY PAPER PLATES)</td>
      <td>0.132653</td>
      <td>0.127551</td>
      <td>0.102041</td>
      <td>0.769231</td>
      <td>6.030769</td>
      <td>1.0</td>
      <td>0.085121</td>
      <td>3.780612</td>
      <td>0.961765</td>
      <td>0.645161</td>
      <td>0.735493</td>
      <td>0.784615</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(SPACEBOY LUNCH BOX )</td>
      <td>(DOLLY GIRL LUNCH BOX)</td>
      <td>0.125000</td>
      <td>0.099490</td>
      <td>0.071429</td>
      <td>0.571429</td>
      <td>5.743590</td>
      <td>1.0</td>
      <td>0.058992</td>
      <td>2.101190</td>
      <td>0.943878</td>
      <td>0.466667</td>
      <td>0.524079</td>
      <td>0.644689</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(DOLLY GIRL LUNCH BOX)</td>
      <td>(SPACEBOY LUNCH BOX )</td>
      <td>0.099490</td>
      <td>0.125000</td>
      <td>0.071429</td>
      <td>0.717949</td>
      <td>5.743590</td>
      <td>1.0</td>
      <td>0.058992</td>
      <td>3.102273</td>
      <td>0.917139</td>
      <td>0.466667</td>
      <td>0.677656</td>
      <td>0.644689</td>
    </tr>
    <tr>
      <th>18</th>
      <td>(SET/20 RED RETROSPOT PAPER NAPKINS )</td>
      <td>(SET/6 RED SPOTTY PAPER CUPS)</td>
      <td>0.132653</td>
      <td>0.137755</td>
      <td>0.102041</td>
      <td>0.769231</td>
      <td>5.584046</td>
      <td>1.0</td>
      <td>0.083767</td>
      <td>3.736395</td>
      <td>0.946471</td>
      <td>0.606061</td>
      <td>0.732362</td>
      <td>0.754986</td>
    </tr>
    <tr>
      <th>19</th>
      <td>(SET/6 RED SPOTTY PAPER CUPS)</td>
      <td>(SET/20 RED RETROSPOT PAPER NAPKINS )</td>
      <td>0.137755</td>
      <td>0.132653</td>
      <td>0.102041</td>
      <td>0.740741</td>
      <td>5.584046</td>
      <td>1.0</td>
      <td>0.083767</td>
      <td>3.345481</td>
      <td>0.952071</td>
      <td>0.606061</td>
      <td>0.701089</td>
      <td>0.754986</td>
    </tr>
    <tr>
      <th>20</th>
      <td>(PLASTERS IN TIN WOODLAND ANIMALS)</td>
      <td>(PLASTERS IN TIN SPACEBOY)</td>
      <td>0.170918</td>
      <td>0.137755</td>
      <td>0.104592</td>
      <td>0.611940</td>
      <td>4.442233</td>
      <td>1.0</td>
      <td>0.081047</td>
      <td>2.221939</td>
      <td>0.934634</td>
      <td>0.512500</td>
      <td>0.549943</td>
      <td>0.685600</td>
    </tr>
    <tr>
      <th>21</th>
      <td>(PLASTERS IN TIN SPACEBOY)</td>
      <td>(PLASTERS IN TIN WOODLAND ANIMALS)</td>
      <td>0.137755</td>
      <td>0.170918</td>
      <td>0.104592</td>
      <td>0.759259</td>
      <td>4.442233</td>
      <td>1.0</td>
      <td>0.081047</td>
      <td>3.443878</td>
      <td>0.898687</td>
      <td>0.512500</td>
      <td>0.709630</td>
      <td>0.685600</td>
    </tr>
    <tr>
      <th>22</th>
      <td>(PLASTERS IN TIN SPACEBOY)</td>
      <td>(PLASTERS IN TIN CIRCUS PARADE )</td>
      <td>0.137755</td>
      <td>0.168367</td>
      <td>0.089286</td>
      <td>0.648148</td>
      <td>3.849607</td>
      <td>1.0</td>
      <td>0.066092</td>
      <td>2.363588</td>
      <td>0.858495</td>
      <td>0.411765</td>
      <td>0.576914</td>
      <td>0.589226</td>
    </tr>
    <tr>
      <th>23</th>
      <td>(PLASTERS IN TIN CIRCUS PARADE )</td>
      <td>(PLASTERS IN TIN SPACEBOY)</td>
      <td>0.168367</td>
      <td>0.137755</td>
      <td>0.089286</td>
      <td>0.530303</td>
      <td>3.849607</td>
      <td>1.0</td>
      <td>0.066092</td>
      <td>1.835747</td>
      <td>0.890096</td>
      <td>0.411765</td>
      <td>0.455263</td>
      <td>0.589226</td>
    </tr>
    <tr>
      <th>24</th>
      <td>(PLASTERS IN TIN WOODLAND ANIMALS)</td>
      <td>(PLASTERS IN TIN CIRCUS PARADE )</td>
      <td>0.170918</td>
      <td>0.168367</td>
      <td>0.102041</td>
      <td>0.597015</td>
      <td>3.545907</td>
      <td>1.0</td>
      <td>0.073264</td>
      <td>2.063681</td>
      <td>0.866000</td>
      <td>0.430108</td>
      <td>0.515429</td>
      <td>0.601538</td>
    </tr>
    <tr>
      <th>25</th>
      <td>(PLASTERS IN TIN CIRCUS PARADE )</td>
      <td>(PLASTERS IN TIN WOODLAND ANIMALS)</td>
      <td>0.168367</td>
      <td>0.170918</td>
      <td>0.102041</td>
      <td>0.606061</td>
      <td>3.545907</td>
      <td>1.0</td>
      <td>0.073264</td>
      <td>2.104592</td>
      <td>0.863344</td>
      <td>0.430108</td>
      <td>0.524848</td>
      <td>0.601538</td>
    </tr>
  </tbody>
</table>
</div>




```python
frequent_itemsets = apriori(basket_Germany, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.sort_values(['lift','support'],ascending=False).reset_index(drop=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>representativity</th>
      <th>leverage</th>
      <th>conviction</th>
      <th>zhangs_metric</th>
      <th>jaccard</th>
      <th>certainty</th>
      <th>kulczynski</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(ROUND SNACK BOXES SET OF 4 FRUITS )</td>
      <td>(ROUND SNACK BOXES SET OF4 WOODLAND )</td>
      <td>0.157549</td>
      <td>0.245077</td>
      <td>0.131291</td>
      <td>0.833333</td>
      <td>3.400298</td>
      <td>1.0</td>
      <td>0.092679</td>
      <td>4.529540</td>
      <td>0.837922</td>
      <td>0.483871</td>
      <td>0.779227</td>
      <td>0.684524</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(ROUND SNACK BOXES SET OF4 WOODLAND )</td>
      <td>(ROUND SNACK BOXES SET OF 4 FRUITS )</td>
      <td>0.245077</td>
      <td>0.157549</td>
      <td>0.131291</td>
      <td>0.535714</td>
      <td>3.400298</td>
      <td>1.0</td>
      <td>0.092679</td>
      <td>1.814509</td>
      <td>0.935072</td>
      <td>0.483871</td>
      <td>0.448887</td>
      <td>0.684524</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(SPACEBOY LUNCH BOX )</td>
      <td>(ROUND SNACK BOXES SET OF4 WOODLAND )</td>
      <td>0.102845</td>
      <td>0.245077</td>
      <td>0.070022</td>
      <td>0.680851</td>
      <td>2.778116</td>
      <td>1.0</td>
      <td>0.044817</td>
      <td>2.365427</td>
      <td>0.713415</td>
      <td>0.251969</td>
      <td>0.577243</td>
      <td>0.483283</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(ROUND SNACK BOXES SET OF4 WOODLAND )</td>
      <td>(SPACEBOY LUNCH BOX )</td>
      <td>0.245077</td>
      <td>0.102845</td>
      <td>0.070022</td>
      <td>0.285714</td>
      <td>2.778116</td>
      <td>1.0</td>
      <td>0.044817</td>
      <td>1.256018</td>
      <td>0.847826</td>
      <td>0.251969</td>
      <td>0.203833</td>
      <td>0.483283</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(ROUND SNACK BOXES SET OF4 WOODLAND )</td>
      <td>(PLASTERS IN TIN WOODLAND ANIMALS)</td>
      <td>0.245077</td>
      <td>0.137856</td>
      <td>0.074398</td>
      <td>0.303571</td>
      <td>2.202098</td>
      <td>1.0</td>
      <td>0.040613</td>
      <td>1.237951</td>
      <td>0.723103</td>
      <td>0.241135</td>
      <td>0.192214</td>
      <td>0.421627</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(PLASTERS IN TIN WOODLAND ANIMALS)</td>
      <td>(ROUND SNACK BOXES SET OF4 WOODLAND )</td>
      <td>0.137856</td>
      <td>0.245077</td>
      <td>0.074398</td>
      <td>0.539683</td>
      <td>2.202098</td>
      <td>1.0</td>
      <td>0.040613</td>
      <td>1.640006</td>
      <td>0.633174</td>
      <td>0.241135</td>
      <td>0.390246</td>
      <td>0.421627</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
