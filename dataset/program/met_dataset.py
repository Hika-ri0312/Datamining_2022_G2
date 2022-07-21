import pandas as pd
import requests
import json
from IPython.display import Image
from PIL import Image
import io
from urllib.request import urlopen
import os

#メトポリ画像のダウンロード

files = os.listdir("./images")
n_files = len(files)

df_met = pd.read_csv('/Users/nakasoneyuuta/info3dm/g2_p/openaccess/MetObjects.csv')

df_met_pd = df_met[df_met['Is Public Domain']==True]
df_met_pd_paint = df_met_pd[df_met_pd['Department'] =='European Paintings']

print(len(df_met_pd_paint))

for i in range(n_files-1, len(df_met_pd_paint)-1):
    obj_id = df_met_pd_paint['Object ID'].iloc[i]
    print(obj_id)

    MET_API = 'https://collectionapi.metmuseum.org/public/collection/v1/objects/'
    target_url = MET_API + str(obj_id)
    res = requests.get(target_url)
    res_dict = json.loads(res.text)
    try:
        file =io.BytesIO(urlopen(res_dict['primaryImage']).read())

    except Exception:
        print("スキップしました")
    else:
        img = Image.open(file)
        img.save(f"./images/{obj_id}.png")