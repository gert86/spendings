from math import ceil
import os
import sys
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
from PyQt5 import QtWidgets as qtw
from requests import head

# 'Qt5Agg' or 'TkAgg'
mpl.use('Qt5Agg')
mpl.rcParams['toolbar'] = 'None'   # no toolbar on bottom

def screen_geometry(monitor):
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
    if backend == "Qt5Agg":
        fig.canvas.manager.window.setGeometry(x, y, w, h)
        fig.canvas.manager.window.statusBar().setVisible(False)
    elif backend == "TkAgg":
        fig.canvas.manager.window.wm_geometry("%dx%d+%d+%d" % (w,h,x,y))
    else:
        print("This backend is not supported yet.")
        print("Set the backend with matplotlib.use(<name>).")
        return

def tile_figures(cols=3, rows=2, monitor=0, screen_rect=None, tile_offsets=None, header_height=0):
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
        header_height:  Height of the window header, might be required for exact placement.
    """    
    assert(isinstance(cols, int) and cols>0)
    assert(isinstance(rows, int) and rows>0)
    assert(screen_rect is None or len(screen_rect)==4)
    backend = mpl.get_backend()
    if screen_rect is None:
        screen_rect = screen_geometry(monitor)
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
    w = int((sw-offX)/cols)
    h_full = int((sh-offY)/rows)
    h = max(0, h_full - header_height)
    for i, num in enumerate(fig_ids):
        fig = plt.figure(num)
        x = (i%cols) *(w) + sx + offX
        y = (i//cols)*(h_full) + sy + offY
        y_full = y + header_height
        set_figure_geometry(fig, backend, x, y_full, w, h)
        #print(f"Figure {i}: x={x}, y_full={y_full}, w={w}, h={h}")

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

def barPlot(title, labels, data, max_y, y_grid_interval, show_bar_labels, color):
  if plt.fignum_exists(title):  
     return
    
  fig = plt.figure(title, layout='constrained')
  x = np.arange(len(labels))
  width = 0.25
  multiplier = 0
  for attribute, measurement in data.items():
    offset = width * multiplier
    rects = plt.bar(x + offset, measurement, width, label=attribute, color=color)
    if show_bar_labels:
      plt.bar_label(rects, padding=3)
    multiplier += 1
  
  plt.xticks(x + width, labels)
  plt.yticks(np.arange(0, max_y*1.2, y_grid_interval))
  plt.grid(axis='y', linestyle='--')
  plt.legend(loc='upper center', ncols=3)
  plt.ylim(0, max_y * 1.2)
  fig.show()   


def createCategoryPlot(category, data_frame_dict, color):
  labels, values = [], []
  for path in csv_file_paths:
    labels.append(os.path.basename(path).split('.')[0].split('_')[-1])
    data_frame = data_frame_dict[path]
    le_sum = data_frame[data_frame['Category']==category]['Amount'].abs().sum()
    values.append(round(le_sum))
  data = {category: values}
  max_y = max_height
  barPlot(category, labels, data, max_y, grid_interval/2, True, color)

def createOverallPlot():
  labels = [os.path.basename(p).split('.')[0].split('_')[-1] for p in csv_file_paths]
  data = {'Incomes': overall_revenue_sums,
          'Spendings': overall_spending_sums}
  max_y = max(overall_spending_sums + overall_revenue_sums)
  barPlot("Overall", labels, data, max_y, grid_interval, False, None)   

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

max_height = config['DEFAULT'].getint('max_height')
grid_interval = config['DEFAULT'].getint('grid_interval')
show_overall_plot = config['DEFAULT'].getboolean('show_overall_plot')

monitor_id = config['DEFAULT']['monitor_id']
monitor_num = config[monitor_id].getint('monitor_num')
tile_offset_x = config[monitor_id].getint('tile_offset_x')
tile_offset_y = config[monitor_id].getint('tile_offset_y')
header_height = config[monitor_id].getint('header_height')


# Get the CSV file path from the command-line argument (specifies a config key)
config_key = "DEFAULT"  # default file
if len(sys.argv) >= 2:
  config_key = sys.argv[1]
csv_file_paths = config[config_key]['csv_file_paths'].split(",")
csv_file_paths = [c.strip("'").strip('"').strip('\n').strip(" ") for c in csv_file_paths]

overall_revenues, overall_spendings = {}, {}
overall_revenue_sums, overall_spending_sums = [], []
for csv_file_path in csv_file_paths:

  # Read the CSV file
  df = pd.read_csv(csv_file_path, names=COLUMNS, sep=';')
  cleanupData(df)

  # Categorize  
  details = {}  
  df['Category'] = df.apply(lambda row: determineCategory(str(row['Subject']), float(row['Amount']), details), axis=1)

  # Filter revenues and spendings
  revenues = df[df['Amount'] > 0]
  overall_revenues[csv_file_path] = revenues
  sum_revenues = round(revenues['Amount'].abs().sum())
  overall_revenue_sums.append(sum_revenues)
  
  spendings = df[df['Amount'] < 0]
  overall_spendings[csv_file_path] = spendings
  sum_spendings = round(spendings['Amount'].abs().sum())
  overall_spending_sums.append(sum_spendings)

  # stacked bars
  fig, ax = plt.subplots()
  plt.title(f"Incomes and Spendings ({os.path.basename(csv_file_path)})")  
  x = ['Incomes '+ str(sum_revenues) + "€", 'Spendings '+ str(sum_spendings) + "€"]  
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
  ax.set_yticks(np.arange(0, max_height+grid_interval, grid_interval))
  ax.grid(axis='y', linestyle='--')
  ax.set(ylim=(0, max_height))
  
  # hover event
  cursor_hover = mplcursors.cursor(pickables=fig, hover=mplcursors.HoverMode.Transient)
  @cursor_hover.connect("add")
  def on_add(sel):
      x, y, width, height = sel.artist[sel.index].get_bbox().bounds      
      x_index = x+width/2      
      text = sel.artist._label
      sel.annotation.set(text=text,
                         anncoords="offset points", ha="left")
      sel.annotation.xy = (x_index, y + height)

  # click event
  cursor_click = mplcursors.cursor(pickables=fig)
  @cursor_click.connect("add")
  def on_add2(sel):
      x, y, width, height = sel.artist[sel.index].get_bbox().bounds      
      x_index = x+width/2
      category = sel.artist._label.split(':')[0]
      sel.annotation.set_visible(False)
      data_frame_dict = overall_revenues if x_index==0 else overall_spendings
      createCategoryPlot(category, data_frame_dict, categories_all[category]["color"])

  # print uncategorized entries (only with single csv) 
  if len(csv_file_paths) == 1:
    uncategorized_revenues = revenues[revenues['Category']==list(categories_revenue.keys())[-1]]
    #print(f"{uncategorized_revenues['Subject']}")
    print(f"{len(uncategorized_revenues)} uncategorized revenues -> {round(uncategorized_revenues['Amount'].abs().sum())}€")

    uncategorized_spendings = spendings[spendings['Category']==list(categories_spending.keys())[-1]]
    print(f"{uncategorized_spendings['Subject'].values}")
    print(f"{len(uncategorized_spendings)} uncategorized spendings -> {round(uncategorized_spendings['Amount'].abs().sum())}€")

  # show current figure
  fig.show()


# arrange all figures in a grid
rows=3
tile_figures(monitor=monitor_num, 
             cols=ceil(len(csv_file_paths)/rows), 
             rows=min(rows, len(csv_file_paths)), 
             tile_offsets=(tile_offset_x,tile_offset_y),
             header_height=header_height)

# show overall figure (optional)
if show_overall_plot and len(csv_file_paths) > 1:
  createOverallPlot()
input("Press enter to quit!")
