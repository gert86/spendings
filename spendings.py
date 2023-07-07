from genericpath import isfile
import os
import sys
from tkinter.tix import REAL
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import mplcursors
import configparser
import json

REAL_CONFIG_PATH = "./data/config.ini"
EXAMPLE_CONFIG_PATH = "./data/example_config.ini"

config = configparser.ConfigParser()
if os.path.isfile(REAL_CONFIG_PATH):
  print(f"\nReading REAL config from {REAL_CONFIG_PATH}\n")
  config.read(REAL_CONFIG_PATH)
else:
  print(f"\nReading EXAMPLE config from {EXAMPLE_CONFIG_PATH}\n")
  config.read(EXAMPLE_CONFIG_PATH)




COLUMNS = ['Konto','Subject','Date1','Date2','Amount','Currency']
results = {}
categories_revenue = json.loads(config['DEFAULT']['categories_revenue'])
categories_spending= json.loads(config['DEFAULT']['categories_spending'])
categories_all = {**categories_revenue, **categories_spending}

details = {}
def determineCategory(subject, amount):
  categories = categories_revenue if amount > 0.0 else categories_spending
  for title,regexes in categories.items():
    for regex in regexes:
      if re.match(regex, subject, re.IGNORECASE):
        if len(str(regex)) > 2:
          details.setdefault(str(regex),[]).append(amount)   # store amounts per regex
        return title
      
  fallback = list(categories_revenue.keys())[-1]
  return fallback
    

def cleanupData(df):
  # Convert the 'Amount' column to numeric data type
  df['Amount'] = pd.to_numeric(df['Amount'].str.replace('.', '').str.replace(',', '.'), errors='coerce')      

  # TODO: consider to remove special characters, etc...to make regex easier, e.g.:
  #df['Subject'] = df.apply(lambda row: row['Subject'].split("|")[-1].strip(), axis=1)

  return df


# Get the CSV file path from the command-line argument (specifies a config key)
config_key = "DEFAULT"  # default file
if len(sys.argv) >= 2:
  config_key = sys.argv[1]

csv_file_path = config[config_key]['csv_file_path']

# Read the CSV file
df = pd.read_csv(csv_file_path, names=COLUMNS, sep=';')
cleanupData(df)

# Categorize
df['Category'] = df.apply(lambda row: determineCategory(str(row['Subject']), float(row['Amount'])), axis=1)

# Filter revenues and spendings
revenues = df[df['Amount'] > 0]
spendings = df[df['Amount'] < 0]

# create figure
x = ['Incomes '+ str(round(revenues['Amount'].abs().sum())) + "€", 
     'Spendings '+ str(round(spendings['Amount'].abs().sum())) + "€"]

rev_heights = []
Ys = []
for cat in categories_revenue:
  le_sum = revenues[revenues['Category']==cat]['Amount'].abs().sum()
  rev_heights.append(round(le_sum))
  Ys.append( [le_sum, 0.0] )
spending_heights = []
for cat in categories_spending:
  le_sum = spendings[spendings['Category']==cat]['Amount'].abs().sum()
  Ys.append( [0.0, le_sum] )
  spending_heights.append(round(le_sum))
# stacked bars
fig, ax = plt.subplots()
offset = np.array([0.0, 0.0])
for y in Ys:
  ax.bar(x, y, bottom = offset, width=0.1)
  offset = np.add(offset, y)
# legend
legendTitles = []
for title in categories_all.keys():
  legendTitles.append(title + "(" + str(round(df[df['Category']==title]['Amount'].abs().sum())) + "€)")
plt.legend(legendTitles)
plt.title(f"Incomes and Spendings ({os.path.basename(csv_file_path)})")

cursor = mplcursors.cursor(hover=mplcursors.HoverMode.Transient)
@cursor.connect("add")
def on_add(sel):
    x, y, width, height = sel.artist[sel.index].get_bbox().bounds
    x_index = x+width/2
    heights = rev_heights if round(x_index)==0 else spending_heights
    categories = categories_revenue if round(x_index)==0 else categories_spending
    stack_index = heights.index(round(height))
    category = list(categories.keys())[stack_index]
    text=f"{category}: {round(height,2)}€\n---------------------------------------------\n"  
    if category in categories.keys():
      regex_list = categories[category]
      for r in regex_list:
        if r in details.keys():
          text += r + ": " + str(round(abs(sum(details[r])), 2)) + "€\n"

    sel.annotation.set(text=text,
                       anncoords="offset points", ha="left")
    sel.annotation.xy = (x_index, y + height)


# print unassigned
uncategorized_revenues = revenues[revenues['Category']==list(categories_revenue.keys())[-1]]
#print(f"{uncategorized_revenues['Subject']}")
print(f"{len(uncategorized_revenues)} uncategorized revenues -> {round(uncategorized_revenues['Amount'].abs().sum())}€")

uncategorized_spendings = spendings[spendings['Category']==list(categories_spending.keys())[-1]]
print(f"{uncategorized_spendings['Subject'].values}")
print(f"{len(uncategorized_spendings)} uncategorized spendings -> {round(uncategorized_spendings['Amount'].abs().sum())}€")

plt.show()
