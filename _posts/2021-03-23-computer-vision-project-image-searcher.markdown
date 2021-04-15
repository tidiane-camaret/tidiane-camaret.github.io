---
layout: post
title:  "Un projet d'analyse d'images : retrouver une photo sur instagram à l'aide d'une description "
date:   2021-03-23 11:20:25 +0100
categories: computer_vision react python data_science
lang: fr
ref: image_searcher
---

Ces dernieres années, plusieurs algorithmes d'analyse d'image, permettant afin d'extraire de l'information d'une image donnée, se sont popularisés. "Extraire de l'information" a ici un sens large : On peut vouloir extraire d'une image des informations basiques, telles que son taux de luminosité,  Mais des méthodes développées ces dernières années permettent de tirer d'une image des informations de plus haut niveau : détecter si l'image contient un visage, si elle contient du texte, ou tel ou tel objet ...

Un champ d'étude de l'analyse d'images, l'image captioning, se penche sur la capacité d'un algorithme à décrire le contenu d'une image, le plus souvent par une phrase plus ou moins longue.

J'ai voulu d'écrire une application mettant en oeuvre un de ces algorithmes, pour me permettre à la fois d'explorer en détail son fonctionnement, mais aussi de comprendre les étapes de construction d'une application : constitution d'une base de données, hébergement sur un serveur, conception d'une interface utilisateur ... 

J'ai décidé de le mettre en application sur une banque de photos bien connue sur internet : Instagram. 

Il semble qu'Instagram applique déja de lui mème, sur chaque image postée par un utilisateur, un algorithme de description d'image. 
Cependant, la description faite semble souvent assez vague :
En plus, il n'est pas possible d'effectuer une recherche sur la description générée.

Nous allons donc voir comment écrire une application permettant, sur un profil donné, de chercher les images correspondant le plus à une description cherchée.



L'application est hébergée ici : <https://photosearch-app.netlify.app/>. Elle est écrire en python pour le côté serveur (back) et en javascript, sous le framework ReactJs, pour le côté front.

# Partie 1 : Trouver des images : Comment fouiller Instagram

Tout d'abord, nous devons trouver un ensemble d'images sur lequel appliquer notre algorithme.
Nous voulons utiliser la plateforme Instagram, nous devons trouver un moyen de parcourir le site et d'en récupérer les photos. En effet, Instagram permet de consulter les images postées par un utilisateur donné, mais ne permet pas de les télécharger d'un coup. A première vue, la seule solution pour se constituer une banque d'images serait donc de parcourir les profils et de les télécharger manuellement une à une.

Heureusement, des programmes permettent de 'scraper' Instagram, c'est à dire de parcourir automatiquement le site afin de récupérer les données qui nous intéressent.

Un de ces programmes, codé en python et nommé instagram-scraper, nous permet de télécharger l'ensemble des photos d'un profil donné. Nous allons essayer ceci sur un profil instagram comportant un grand nombre d'images diverses, un compte Instagram de la chaine National Geographic.

le script est disponible sur github : <https://github.com/arc298/instagram-scraper>.

```console
$pip install instagram-scraper
$instagram-scraper natgeotravel  
```

Le téléchargement prend un temps assez long, environ 10 photos par seconde. Heureusement, le script possède un option permettant de récupérer uniquement les adresses url des images, avec un temps de téléchargement potentiellement plus rapide. Ces adresses nous suffirons pour accéder aux images plus tard.

```console
$instagram-scraper natgeotravel --media-metadata --media-types none
```

Le temps d'execution est effectivement bien plus court.


# Partie 2 : Décrire une image : Comment analyser une image afin d'en tirer une description 

Maintenant que nous avons accès aux images, nous allons pouvoir assigner à chaqune d'entre elles une description. 
Une méthode traditionnelle de machine learning pour extraire les informations d'une image est l'utilisation d'un réseau de neurones, et plus particulièrement les réseau de neurones convolutif (CNN). Il applique des convolutions sur la surface de l'image, permettant d'y détecter des motifs de plus ou moins haut niveau, et d'associer ces motifs aux caratéristiques de tel ou tel objet. 

Lors de l'entrainement du réseau, celui-ci apprendra à la fois les caractéristiques des convolutions à appliquer, et à quel contenu associer ces motifs : visage, voiture, chat, etc ...

Des CNN déja entrainés sur de grandes bases de données d'images sont disponibles sur Internet. Parmi eux, [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), un CNN entrainé sur la base d'images [ImageNet](http://www.image-net.org/) (1.2 millions d'images de 1000 classes différentes), et publiée en 2012, reste encore aujourd'hui un référence de modèle pré-entrainé.


Celui-ci est disponible au téléchargement sur la [page Kaggle](https://www.kaggle.com/pytorch/alexnet) de la librarie python Pytorch. 

Il peut être téléchargé directement depuis un script python :

{% highlight python %}
import torch
model = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
{% endhighlight %}

Nous pouvons alors récupérer l'image de notre choix via son url, via les librairies PIL et urllib :

{% highlight python %}
from PIL import Image
from urllib.request import urlopen
input_image = Image.open(urlopen("https://www.nosamis.fr/img/classic/Berger-Australien1.jpg"))
{% endhighlight %}

Nous normalisons l'image et la redimensionnons, puis la mettons sous forme de tenseur : 
{% highlight python %}
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
{% endhighlight %}

Nous faisons passer notre tenseur dans le modèle AlexNet :

{% highlight python %}
with torch.no_grad():
    output = model(input_batch)
{% endhighlight %}

le tenseur `output` contient les scores associés aux 1000 objets que peut reconnaitre AlexNet. Nous appliquons la fonction softmax aux scores pour en obtenir les probabilités : 

{% highlight python %}
probabilities = torch.nn.functional.softmax(output[0], dim=0)
{% endhighlight %}

Nous pouvons obtenir la liste des noms des 1000 objects reconnus par AlexNet sur le github de Pytorch : 

```console
$wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```
```python
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

categories[torch.argmax(probabilities)] #resultat : 'Border collie'
```
Cette méthode est rapide, et surtout précise sur les images que nous lui donnons en exemple :

Cependant, elle nous fournit uniquement une liste d'éléments probables contenus dans l'image. Elle n'extrait pas d'actions telles que "courir", "faire du vélo", "allongé", ni d'éléments contextuels de l'image tels que "sur un banc", "au bord d'une plage", etc ...  

Nous souhaiterions, pour chaque image, produire une phrase résumant son contenu, et ceci qu'elle possède des éléments identifiables individuellement ou non.

Le papier [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044), de Kelvin Xu et Al propose une architecture permettant d'associer à une image une phrase descriptive. Celle-ci est constituée d'un CNN, ainsi que d'un réseau récurrent (RNN) permettant de générer des phrases cohérentes et en rapport avec l'image.

Nous utiliserons ici une implémentation écrite par [Sgrvinod](https://github.com/sgrvinod), et disponible [ici](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning), que nous décrirons dans un prochain article.

```console
$python3 imcap_sgrvinod/caption_photosearch.py  --model='imcap_sgrvinod/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='imcap_sgrvinod/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5 --imglist='url_list_natgeotravel.json'--dict_path='captions_dict_natgeotravel.json'
```

Ce script ouvre chaque url présent dans le dictionnaire contenu dans `url_list_natgeotravel.json`, et écrit les descriptions produites par le modèle dans `captions_dict_natgeotravel.json`

Nous avons donc une phrase descriptive de chaque photo du compte NatGeoTravel. Nous allons nous en servir pour construire un moteur de recherche sur ces photos.


# Partie 3 : Récupérer les meilleurs résultats : créer une API renvoyant les images les plus proches d'une descrption donnée

Notre but est de récupérer, pour une requète donnée, les *n*
images dont les descriptions sont les plus proches de la requète. 


```python
documents = [d["caption"] for d in name_data]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    # Convert the X as transposed matrix
    X = X.T.toarray()# Create a DataFrame and set the vocabulary as the index
    df = pd.DataFrame(X, index=vectorizer.get_feature_names())

```

```python
 q = [search_string]
    q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
    sim = []  # Calculate the similarity
    
    for i in range(len(documents)):
        sim.append(np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec))

```

Nous trions notre liste d'urls par ordre croissant de degré de similitude, puis retenons les 100 premiers. 

```python
    paired_scores = zip(sim,urls,txts)

    #sorted = [x for _,x in sorted(paired_scores)]
    sorted_scores = sorted(paired_scores, key = lambda t : t[0],reverse = True)
    # Create an empty list for our results

    results = [i for i in sorted_scores if i[0] > 0]

    if len(results) > 100 :
        results = results[:100]
```


Notre but final est de créer une application permettant à n'importe quel utilisateur de taper sa requète, puis d'obtenir les résultats fournis par ce script.
Nous allons donc faire en sorte que le script ci-dessus se déclenche à chaque requète, en l'inscrivant à l'interieur d'un serveur. Ceci est permis par le framework Flask : 

```python
import flask
from flask import request, jsonify
from fuzzywuzzy import fuzz
import json


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


app = flask.Flask(__name__)
app.config["DEBUG"] = True

data_file = open('photosearch_db.json', 'r') 
data = json.load(data_file)

@app.route('/api/v1/search', methods=['GET'])
def api_id():
    # Check if a search string and name were provided.
    if ('str' in request.args) and ('name' in request.args):
        search_string = str(request.args['str'])
        name = str(request.args['name'])
    else:
        return "Error: No search string or name provided. Please specify a string."

    if name in data : 
    	name_data = data[name]
    else : 
    	return "Error: This name is not in the database. A future version will allow to look for new names."




    documents = [d["caption"] for d in name_data]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    # Convert the X as transposed matrix
    X = X.T.toarray()# Create a DataFrame and set the vocabulary as the index
    df = pd.DataFrame(X, index=vectorizer.get_feature_names())


    q = [search_string]
    q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
    sim = []  # Calculate the similarity
    
    for i in range(len(documents)):
        sim.append(np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec))
  
    urls = []
    txts = []

    for img in name_data :

        urls.append(img["url"])
        txts.append(img["caption"] + img["yolo_objects"])


    paired_scores = zip(sim,urls,txts)

    #sorted = [x for _,x in sorted(paired_scores)]
    sorted_scores = sorted(paired_scores, key = lambda t : t[0],reverse = True)
    # Create an empty list for our results

    results = [i for i in sorted_scores if i[0] > 0]

    if len(results) > 100 :
        results = results[:100]

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    

    response = jsonify(results)

    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response


@app.route('/')
def index():
    return "<h1>Le serveur marche !!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
```

Nous pouvons lancer notre serveur sur 


# Partie 4 : Afficher les résultats : créer une application interagissant avec l'API