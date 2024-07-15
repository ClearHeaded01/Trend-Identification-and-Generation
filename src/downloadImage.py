import urllib.request
import pandas as pd

path = "<enter path to folder>"
styles_df = pd.read_csv("Fashion dataset.csv")
img_path = path + '\\train_images\\'

req = urllib.request.build_opener()
req.addheaders = [('User-Agent', 'Mozilla/5.0'), ('Accept','text/html'),]
urllib.request.install_opener(req)

# styles_df['img_id'] = [i for i in range(0,styles_df.shape[0])]

def downloadImage():
    for i in range(0,styles_df.shape[0]):
        if(styles_df.at[i,'img'] == None or styles_df.at[i,'img'] == 'nan'):
            print("This line is empty: ", i)
            break
        filename = img_path + str(styles_df.at[i,'img_id'])
        url = styles_df.at[i,'img']
        loc = urllib.request.urlretrieve(url, filename)
        print(loc)
downloadImage()
