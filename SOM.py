import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rnd
from matplotlib import patches
import geopandas as gpd


#normalizing the data attributes
df = pd.read_csv('Data_Happinness.csv')
df = df.dropna()
exclude_cols = [0, 1] 
cols_to_normalize = df.columns.difference(df.columns[exclude_cols])
df_norm = (df[cols_to_normalize] - df[cols_to_normalize].min()) / (df[cols_to_normalize].max() - df[cols_to_normalize].min())
df_new = pd.concat([df[df.columns[exclude_cols]], df_norm], axis=1)
df_new.to_csv('Happiness_Data_normalized.csv', index=False)

#norm_weights to colours
RGB_dict = dict()
colour_map = [(255/255.0,192/255.0,203/255.0),(150/255.0,75/255.0,0),(0,255/255.0,0),(255/255.0,165/255.0,0),(255/255.0,255/255.0,0),(127/255.0,0,255/255.0)]
data = open('Happiness_Data_normalized.csv','r')
for line in  data.readlines()[1:]:
    line = line.split(',')
    if line[1] == '2019':
        country = line[0]
        values = []
        for n in line[2:]:
          values.append(float(n))
        rgb_vals = [tuple(c * v for c, v in zip(color, values)) for color in colour_map]
        RGB_dict[country] = tuple(map(lambda x: sum(x)/len(x), zip(*rgb_vals)))
input_mat = []
for keys in RGB_dict.keys():
    val_mat = []
    for val in RGB_dict[keys]:
        val_mat.append(val)
    input_mat.append(val_mat)
input_mat = np.array(input_mat)
print(input_mat)

#initializing SOM
n = 12
m = 12
SOM = np.random.rand(m,n,3)

def find_BMU(input_vec,SOM):
    distances = np.sum(np.abs(SOM - input_vec), axis=-1)
    bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
    return np.array(bmu_index)

def find_neighbourhood(SOM_coords,BMU_coords,radius):
    dist = np.sum(np.square(np.array(BMU_coords)-np.array(SOM_coords)))
    return np.exp(-dist / (2*(radius**2)))

def update_weights(SOM,learning_rate,radius,BMU_coords,input_vec,step=3):
    if radius < 1e-3:
        delta_weight = learning_rate*(input_vec - SOM[BMU_coords[0],BMU_coords[1]])
        SOM[BMU_coords[0],BMU_coords[1]] += delta_weight
    else:
        for x in range(max(0, BMU_coords[0]-step), min(SOM.shape[0], BMU_coords[0]+step+1)):
            for y in range(max(0, BMU_coords[1]-step), min(SOM.shape[1], BMU_coords[1]+step+1)):
                SOM[x, y]  += find_neighbourhood([x,y],bmu_index,radius)*learning_rate_0*(input_vec-SOM[x,y])
    return SOM

def show_map(epoch):
    fig, ax = plt.subplots()
    im = ax.imshow(SOM)
    ax.set_title('Self-Organising Map at epoch  %d' % epoch)
    for i, country in enumerate(RGB_dict.keys()):
        BMU_index = find_BMU(RGB_dict[country], SOM)
        x, y = BMU_index[1], BMU_index[0]
        ax.annotate(country[0:3], xy=(x, y), xytext=(x + 0.1, y + 0.1),
                color='black', fontsize=8,ha='center', va='center')
    num_rows, num_cols, _ = SOM.shape
    ax.set_xticks(np.arange(num_cols+1)-0.5, minor=True)
    ax.set_yticks(np.arange(num_rows+1)-0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    plt.show()

#initializing constant values
learning_rate_0 = 0.08
radius_0 = 6
epochs = 10000
T_cons_r = epochs/np.log(radius_0)
T_cons_lr = epochs/np.log(learning_rate_0)
learning_rate = learning_rate_0
radius = radius_0

for epoch in range(epochs):
    vector = input_mat[np.random.randint(0,input_mat.shape[0]), :]
    bmu_index = find_BMU(vector, SOM)
    SOM = update_weights(SOM,learning_rate,radius,bmu_index,vector)
    learning_rate = learning_rate_0 * np.exp(-epoch / T_cons_lr)
    radius = radius_0 * np.exp(-epoch / T_cons_r)
    if epoch%1000 == 0:
        show_map(epoch)
show_map(epochs)

#displaying on the world map:
worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
worldmap.plot()
plt.show()

#creating a dictionary of BMU_values for each country:
for country in RGB_dict.keys():
    BMU_index =find_BMU(RGB_dict[country],SOM)
    RGB_dict[country] = SOM[BMU_index[0],BMU_index[1]]

df = pd.DataFrame.from_dict(RGB_dict, orient='index', columns=['R','G','B'])
merged = worldmap.merge(df, left_on='name', right_index=True)
print(merged.head())
merged.plot(column=['R','G','B'])
plt.show()