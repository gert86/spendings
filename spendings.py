from genericpath import isfile
from math import ceil
import os
import sys
from tkinter.tix import REAL
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import numpy as np
import mplcursors
import configparser
import json

mpl.use('TKAgg')
mpl.rcParams['toolbar'] = 'None'   # no toolbar on bottom

def screen_geometry(monitor=0):
    try:
        from screeninfo import get_monitors
        sizes = [(s.x, s.y, s.width, s.height) for s in get_monitors()]
        return sizes[monitor]
    except ModuleNotFoundError:
        default = (0, 0, 900, 600)
        print("screen_geometry: module screeninfo is no available.")
        print("Returning default: %s" % (default,))
        return default

def set_figure_geometry(fig, backend, x, y, w, h):
    if backend in ("Qt5Agg", "Qt4Agg"):
        fig.canvas.manager.window.setGeometry(x, y, w, h)
        #fig.canvas.manager.window.statusBar().setVisible(False)
        #fig.canvas.toolbar.setVisible(True)
    elif backend in ("TkAgg",):
        fig.canvas.manager.window.wm_geometry("%dx%d+%d+%d" % (w,h,x,y))
    else:
        print("This backend is not supported yet.")
        print("Set the backend with matplotlib.use(<name>).")
        return

def tile_figures(cols=3, rows=2, screen_rect=None, tile_offsets=None):
    """
    Tile figures. If more than cols*rows figures are present, cols and
    rows are adjusted. For now, a Qt- or Tk-backend is required.

        import matplotlib
        matplotlib.use('Qt5Agg')
        matplotlib.use('TkAgg')

    Arguments: 
        cols, rows:     Number of cols, rows shown. Will be adjusted if the 
                        number of figures is larger than cols*rows.
        screen_rect:    A 4-tuple specifying the geometry (x,y,w,h) of the 
                        screen area used for tiling (in pixels). If None, the 
                        system's screen is queried using the screeninfo module.
        tile_offsets:   A 2-tuple specifying the offsets in x- and y- direction.
                        Can be used to compensate the title bar height.
    """    
    assert(isinstance(cols, int) and cols>0)
    assert(isinstance(rows, int) and rows>0)
    assert(screen_rect is None or len(screen_rect)==4)
    backend = mpl.get_backend()
    if screen_rect is None:
        screen_rect = screen_geometry()
    #print(f"screen_rect: {screen_rect}")
    #print(f"rows={rows}, cols={cols}")
    if tile_offsets is None:
        tile_offsets = (0,0)
    sx, sy, sw, sh = screen_rect
    offX = tile_offsets[0]
    offY = tile_offsets[1]
    #print(f"sx={sx}, sy={sy},offX={offX}, offY={offY}")
    fig_ids = plt.get_fignums()
    # Adjust tiles if necessary.
    tile_aspect = cols/rows
    while len(fig_ids) > cols*rows:
        cols += 1
        rows = max(np.round(cols/tile_aspect), rows)
    # Apply geometry per figure.
    w = int(sw/cols) - offX
    h = int(sh/rows) - offY
    for i, num in enumerate(fig_ids):
        fig = plt.figure(num)
        x = (i%cols) *(w+offX) + sx + offX    #(offX if (i)%cols!=0 else 0)
        y = (i//cols)*(h+offY) + sy + offY    #(offY if i >= rows*cols else 0)
        set_figure_geometry(fig, backend, x, y, w, h)
        #print(f"Figure {i}: x={x}, y={y}, w={w}, h={h}")

def testTiling(n_figs=10, backend="TKAgg", **kwargs):
    mpl.use(backend)
    plt.close("all")
    for i in range(n_figs):
        plt.figure()
    tile_figures(**kwargs)
    plt.show()

def determineCategory(subject, amount, details_container):
  categories = categories_revenue if amount > 0.0 else categories_spending
  for title,colorRegexTuple in categories.items():
    regexes = colorRegexTuple["regex"]
    for regex in regexes:
      if re.match(regex, subject, re.IGNORECASE):
        if len(str(regex)) > 2:
          details_container.setdefault(str(regex),[]).append(amount)   # store amounts per regex
        return title      
  fallback = list(categories_revenue.keys())[-1]
  return fallback
    
def cleanupData(df):
  # Convert the 'Amount' column to numeric data type
  df['Amount'] = pd.to_numeric(df['Amount'].str.replace('.', '', regex=False)
                                           .str.replace(',', '.', regex=False), errors='coerce')      

  # TODO: consider to remove special characters, etc...to make regex easier, e.g.:
  #df['Subject'] = df.apply(lambda row: row['Subject'].split("|")[-1].strip(), axis=1)

  return df

def getSumAndDetailText(data_frame, cat):
  le_sum = data_frame[data_frame['Category']==cat]['Amount'].abs().sum()
  text=f"{cat}: {round(le_sum,2)}€\n---------------------------------------------\n"
  for r in categories_all[cat]["regex"]:
    if r in details.keys():
      text += r + ": " + str(round(abs(sum(details[r])), 2)) + "€\n"   
  return le_sum, text



##############################################################################################
#      MAIN
##############################################################################################
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

# Get the CSV file path from the command-line argument (specifies a config key)
config_key = "DEFAULT"  # default file
if len(sys.argv) >= 2:
  config_key = sys.argv[1]

csv_file_paths = config[config_key]['csv_file_paths'].split(",")
csv_file_paths = [c.strip("'").strip('"').strip('\n').strip(" ") for c in csv_file_paths]
for csv_file_path in csv_file_paths:

  # Read the CSV file
  df = pd.read_csv(csv_file_path, names=COLUMNS, sep=';')
  cleanupData(df)

  # Categorize  
  details = {}  
  df['Category'] = df.apply(lambda row: determineCategory(str(row['Subject']), float(row['Amount']), details), axis=1)

  # Filter revenues and spendings
  revenues = df[df['Amount'] > 0]
  spendings = df[df['Amount'] < 0]

  # stacked bars
  fig, ax = plt.subplots()
  plt.title(f"Incomes and Spendings ({os.path.basename(csv_file_path)})")
  x = ['Incomes '+ str(round(revenues['Amount'].abs().sum())) + "€", 
       'Spendings '+ str(round(spendings['Amount'].abs().sum())) + "€"]  
  offset = np.array([0.0, 0.0])
  for cat,data in categories_revenue.items():
     le_sum, text = getSumAndDetailText(revenues, cat)
     if le_sum != 0.0:
        ax.bar(x, [le_sum, 0.0], bottom = offset, width=0.1, color=data["color"], label=text)
        offset = np.add(offset, [le_sum, 0.0])
  for cat,data in categories_spending.items():
     le_sum, text = getSumAndDetailText(spendings, cat)
     if le_sum != 0.0:
        ax.bar(x, [0.0, le_sum], bottom = offset, width=0.1, color=data["color"], label=text)
        offset = np.add(offset, [0.0, le_sum])           

  # legend
  legendHandles = []
  for title in categories_all.keys():
    amount = round(df[df['Category']==title]['Amount'].abs().sum())
    if amount != 0:
      color = categories_all[title]["color"]
      label = title + "(" + str(amount) + "€)"
      legendHandles.append(mpatches.Patch(color=color, label=label))
  ax.legend(handles=legendHandles, prop={'size': 6})

  # grid
  max_y = 50000
  grid_interval = 10000
  ax.set_yticks(np.arange(0, max_y+grid_interval, grid_interval))
  ax.grid(axis='y', linestyle='--')
  ax.set(ylim=(0, max_y))
  fig.canvas.manager.window.overrideredirect(1)  # remove window frame

  cursor = mplcursors.cursor(hover=mplcursors.HoverMode.Transient)
  @cursor.connect("add")
  def on_add(sel):
      x, y, width, height = sel.artist[sel.index].get_bbox().bounds      
      x_index = x+width/2      
      text = sel.artist._label
      sel.annotation.set(text=text,
                         anncoords="offset points", ha="left")
      sel.annotation.xy = (x_index, y + height)


  # print uncategorized entries (only with single csv) 
  if len(csv_file_paths) == 1:
    uncategorized_revenues = revenues[revenues['Category']==list(categories_revenue.keys())[-1]]
    #print(f"{uncategorized_revenues['Subject']}")
    print(f"{len(uncategorized_revenues)} uncategorized revenues -> {round(uncategorized_revenues['Amount'].abs().sum())}€")

    uncategorized_spendings = spendings[spendings['Category']==list(categories_spending.keys())[-1]]
    print(f"{uncategorized_spendings['Subject'].values}")
    print(f"{len(uncategorized_spendings)} uncategorized spendings -> {round(uncategorized_spendings['Amount'].abs().sum())}€")

  fig.show()


# arrange all figures in a grid
rows=3
tile_figures(cols=ceil(len(csv_file_paths)/rows), rows=min(rows, len(csv_file_paths)), tile_offsets=(0,0))
input("Press any key to quit!")
