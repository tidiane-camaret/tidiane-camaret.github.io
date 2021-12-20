---
layout: post
title:  "Un projet d'analyse d'images : retrouver une photo sur instagram à l'aide de sa description"
date:   2021-03-23 11:20:25 +0100
categories: computer_vision react python data_science
lang: fr
ref: image_searcher
---

La plupart des sites de partage d'images tirent profit d'algorithmes pour labelliser et classer rapidement les images postées. Instagram [applique déja de lui mème, sur chaque photo postée par un utilisateur, des méthodes de classification d'images.](https://www.theverge.com/2018/11/28/18116323/instagram-ai-visual-impairment-description)
Mais quand on examine ces descriptions, elles sont souvent assez vagues :

![img2](/assets/images/project_image_searcher/im2.png)

![img3](/assets/images/project_image_searcher/im3.png)



![img4](/assets/images/project_image_searcher/im4.png)

![img5](/assets/images/project_image_searcher/im5.png)

En plus de ça, Instagram ne propose pas d'outil de recherche sur les descriptions générées.

On pourrait appliquer sur ces photos d'autres méthodes d'analyse d'image, permettant de les décrire plus précisément. Ca nous permettrait de trouver facilement une image à l'aide d'une description.

On va donc ici décrire le fonctionnement d'une démo web permettant d'effectuer une recherche d'image sur un profil Instagram, à partir de descriptions que nous générerons numériquement. On utilisera **python** pour le côté serveur (back) et **javascript** pour le front.

# Partie 1 : Trouver des images : Comment fouiller Instagram

On a d'abord besoin d'un ensemble d'images sur lequel effectuer notre recherche. Comme nous voulons utiliser la plateforme Instagram, on doit trouver un moyen de parcourir le site et d'en récupérer le contenu.

Instagram ne propose pas d'interface pour télécharger plusieurs photos d'un coup. A première vue, la seule solution pour se constituer une banque d'images serait donc de parcourir les profils et de les télécharger manuellement une à une.

Heureusement, des programmes permettent de **'scraper'** Instagram, c'est-à-dire de parcourir automatiquement le site afin de récupérer les données qui nous intéressent.

Un de ces programmes, codé en python par [arc298](https://github.com/arc298/) et nommé [instagram-scraper](https://github.com/arc298/instagram-scraper), nous permet de télécharger l'ensemble des photos d'un profil. Nous allons essayer ceci sur un profil instagram comportant un grand nombre d'images diverses : le compte **NatGeoTravel** de la chaîne National Geographic.

```console
$pip install instagram-scraper
$instagram-scraper natgeotravel  
```

Le téléchargement prend un temps assez long, environ 10 photos par seconde. Heureusement, le script possède une option permettant de récupérer uniquement les adresses url des images, avec un temps de téléchargement potentiellement plus rapide. Ces adresses nous suffiront pour accéder aux images plus tard.

```console
$instagram-scraper natgeotravel --media-metadata --media-types none
```

Le temps d'exécution est effectivement bien plus court.
On stocke la liste d'urls au format json, dans un fichier nommé `url_list_natgeotravel.json`. Celui-ci va constituer notre base de données.

![img6](/assets/images/project_image_searcher/im6.png)
*Liste des urls extraites.*

# Partie 2 : Décrire une image : Comment analyser une image afin d'en tirer une description

Maintenant qu'on a accès aux images, on va pouvoir assigner à chacune d'entre elles une description.
Une méthode traditionnelle de machine learning pour extraire les informations d'une image est l'utilisation d'un **réseau de neurones**, et plus particulièrement les réseaux de neurones convolutifs (CNN). Il applique des convolutions sur la surface de l'image, permettant d'y détecter des motifs de plus ou moins haut niveau, et d'associer ces motifs aux caractéristiques de tel ou tel objet.

Lors de l'entraînement du réseau, celui-ci apprendra à la fois les caractéristiques des convolutions à appliquer, et **à quel contenu associer ces motifs** : visage, voiture, chat, etc ...

![img7](/assets/images/project_image_searcher/im7.png)

*Un schéma d'architecture possible pour un CNN* (source : [Kdnuggets.com](https://www.kdnuggets.com/2016/11/intuitive-explanation-convolutional-neural-networks.html/3))

Des CNN déjà entraînés sur de grandes bases de données d'images sont disponibles sur Internet. Parmi eux, [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), un CNN entrainé sur la base d'images [ImageNet](http://www.image-net.org/) (1.2 millions d'images de 1000 classes différentes), et publié en 2012, reste aujourd'hui un référence de modèle pré-entrainé.

![img8](/assets/images/project_image_searcher/im8.png)

*Quelques échantillons d'ImageNet* (source : [Devopedia](https://devopedia.org/imagenet) )

AlexNet est disponible au téléchargement sur la [page Kaggle](https://www.kaggle.com/pytorch/alexnet) de la librarie python *Pytorch*.

Il peut également être téléchargé directement depuis un script python :

{% highlight python %}
import torch
model = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
{% endhighlight %}

On peut alors récupérer l'image de notre choix via son url, avec les librairies *PIL* et *urllib* :

{% highlight python %}
from PIL import Image
from urllib.request import urlopen
input_image = Image.open(urlopen("https://www.nosamis.fr/img/classic/Berger-Australien1.jpg"))
{% endhighlight %}

![img9](/assets/images/project_image_searcher/im9.jpg)

*Notre image test : un berger australien.*

Nous normalisons l'image et la redimensionnons, puis la mettons sous forme de tenseur, un tableau multidimensionnel utilisé comme format de données par la librairie Pytorch :

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

Le tenseur `output` contient les scores associés aux 1000 objets que peut reconnaître AlexNet. Nous appliquons la fonction softmax aux scores pour en obtenir les probabilités :

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

La probabilité la plus haute sortie par AlexNet est celle de la classe Border collie, ce qui n'est pas exactement le contenu de notre photo initiale, mais tout de même très proche compte tenu de la diversité des classes proposées par le modèle.

Cette méthode est précise sur les exemples d'images que nous lui présentons :


![img10](/assets/images/project_image_searcher/im10.png)

![img11](/assets/images/project_image_searcher/im11.jpg)

![img12](/assets/images/project_image_searcher/im12.jpg)

Malheureusement, elle nous fournit uniquement une liste d'éléments probables contenus dans l'image. Elle n'extrait pas d'actions telles que "courir", "faire du vélo", "allongé", ni d'éléments contextuels de l'image tels que "sur un banc", "au bord d'une plage", etc ...  

On aimerait, pour chaque image, produire une phrase résumant son contenu, et ceci qu'elle possède des éléments identifiables individuellement ou non.

Le papier [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044), de Kelvin Xu et Al propose une architecture permettant d'associer à une image une phrase descriptive. Celle-ci est constituée d'un CNN, ainsi que d'un réseau récurrent (RNN) permettant de générer des phrases cohérentes et en rapport avec l'image. 

On va utiliser ici un script librement inspiré d'une implémentation de [Sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning), que je décris [dans cet article.](http://127.0.0.1:4000/computer_vision/react/python/data_science/2021/04/18/computer-vision-image-captioning-fr.html)

```console
$python3 imcap_sgrvinod/caption_photosearch.py  --model='imcap_sgrvinod/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='imcap_sgrvinod/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5 --imglist='url_list_natgeotravel.json'--dict_path='captions_dict_natgeotravel.json'
```

Ce script ouvre chaque url présent dans le dictionnaire contenu dans `url_list_natgeotravel.json`, et écrit les descriptions produites par le modèle dans `captions_dict_natgeotravel.json`
![img1](/assets/images/project_image_searcher/im1.png)
*Exemple d'une description d'image générée par le script.*

On a donc maintenant une phrase descriptive pour chaque photo du compte NatGeoTravel. On va s'en servir pour construire un moteur de recherche sur ces photos :


# Partie 3 : Récupérer les meilleurs résultats : créer une API renvoyant les images les plus proches d'une description donnée

Notre but est de récupérer, pour une requête donnée, les *n* images dont les descriptions sont les plus proches de la requête. On va pour ça utiliser une méthode standard de recherche de texte, la méthode **TF-IDF**, basée sur les fréquences d'apparition des termes de la requête dans les documents recherchés.

Imaginons que nous voulons retrouver une image de personnes marchant dans la rue. Notre requète sera la chaîne de caractères suivante : "People walking down the street".
Chaque description peut contenir un nombre plus ou moins important de termes cherchés, ici en **gras** :


| image |                     sentence                    |
|-------|:-----------------------------------------------:|
| img1  |            A dog **walking** on a **street**            |
| img2  |        A young girl surfing in **the** ocean        |
| img3  |    A group of **people walking down the street**    |
| img4  | A couple of **people** playing football in **the** park |


La méthode **TF-IDF** se base sur l'hypothèse que deux documents partageant un grand nombre de mots ont des chances d'être proches sémantiquement. Nous calculerons le degré de similarité entre notre requête et un texte j par le nombre de fois où un terme de la requête apparaît dans j.

Afin d'obtenir un degré de similarité entre 0 et 1, on divise (ou **normalise** ce nombre par le produit du nombre de mots de la requête et du texte j.
*Cette normalisation nous permet également en théorie de pénaliser les textes plus longs, qui de par leur taille ont statistiquement plus de chance de contenir des mots de la requête. Cependant, étant donné que les textes cherchés ont ici ont tous une taille similaire, cette normalisation a peu d'impact dans notre cas.*

On utilise ici la fonction TfidfVectorizer() de la bibliothèque *sklearn*, qui nous permet de transformer facilement nos textes en tableaux d'occurrence de mots :


```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    import numpy as np

    documents = [d["caption"] for d in name_data]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    # Convert the X as transposed matrix
    X = X.T.toarray()# Create a DataFrame and set the vocabulary as the index
    df = pd.DataFrame(X, index=vectorizer.get_feature_names())

```

On calcule les degrés de similarité entre chaque texte et la requête, puis les ajoutons à notre liste d'urls.

```python
 q = [search_string]
    q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
    sim = []  # Calculate the similarity
    
    for i in range(len(documents)):
        sim.append(np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec))

    paired_scores = zip(sim,urls,txts)

```

On trie notre liste d'urls par ordre croissant de degré de similarité, puis on retient les 100 premiers.

```python

    sorted_scores = sorted(paired_scores, key = lambda t : t[0],reverse = True)
    # Create an empty list for our results

    results = [i for i in sorted_scores if i[0] > 0]

    if len(results) > 100 :
        results = results[:100]
```


Notre but est de créer une application permettant à n'importe quel utilisateur de taper sa requête, puis d'obtenir les résultats fournis par ce script.
On va donc faire en sorte que le script ci-dessus se déclenche à chaque requête, en l'inscrivant à l'intérieur d'un serveur. Nous utilisons ici le framework [Flask](https://flask.palletsprojects.com/en/1.1.x/).

Pour lancer notre serveur local Flask, écrivons le script suivant, qu'on appelera `api.py` :

```python
import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/')
def index():
    return "<h1>Le serveur marche !!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)

```

On a configuré notre serveur pour qu'il nous renvoie le message html "*Le serveur marche !!*" lorsque nous consultons sa racine.

On lance notre script :
```console
$python3 api.py
```

Notre serveur tourne maintenant en local à l'adresse http://127.0.0.1:5000/ :

![img13](/assets/images/project_image_searcher/im13.png)

On va modifier notre script pour qu'à réception d'une requête, il nous renvoie les urls classées.

On introduit quelques librairies utiles :
```python
import flask
from flask import request, jsonify
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


app = flask.Flask(__name__)
app.config["DEBUG"] = True
```

On ouvre ensuite notre base de données :
```python
data_file = open('photosearch_db.json', 'r')
data = json.load(data_file)
```

On récupère les arguments 'name' et 'search_string', qui seront respectivement l'identifiant Instagram sur lequel faire la recherche, et le texte cherché :
```python
@app.route('/', methods=['GET'])
def api_id():
    # Check if a search string and name were provided.
    if ('str' in request.args) and ('name' in request.args):
        search_string = str(request.args['str'])
        name = str(request.args['name'])

```
On calcule les 100 meilleurs résultats selon la méthode **TF-IDF** et renvoyons les résultats :

```python

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
```

On peut maintenant tester notre serveur, avec la requète "people walking on the street", sur le compte "natgeotravel", en accédant à l'adresse  http://127.0.0.1:5000/?str=people+walking+on+street&name=natgeotravel :

![img15](/assets/images/project_image_searcher/im15.png)

Le serveur nous renvoie bien une liste de 100 urls, ainsi que les scores de similarités et descriptions correspondantes.


# Partie 4 : Afficher les résultats : créer une application interagissant avec l'API

On va maintenant construire une interface web nous permettant d'envoyer nos requêtes à l'API, et d'afficher les résultats reçus.

Plusieurs maquettes d'interface sont disponibles sur Github. On va utiliser une [maquette javascript](https://github.com/lelouchB/react-photo-search) codée par [lelouchB](https://github.com/lelouchB), sous le framework react.

En l'état, l'application utilise l'API d'[Unsplash](https://unsplash.com/developers), une banque d'images. On va modifier le code source du fichier `searchPhotos.js` pour qu'elle utilise notre API.

On commence d'abord par importer `axios`, une librairie facilitant la gestion des requêtes http :

```javascript
import React, { Fragment, useState, useEffect } from 'react';
import axios from 'axios';
```

On écrit ensuite nos hooks, qui définiront l'état de nos requêtes :
```javascript
export default function SearchInst() {
  const [data, setData] = useState([]);
  const [query, setQuery] = useState('people walking on street');
  const [name, setName] = useState('natgeotravel');
  const [url, setUrl] = useState(
    'http://127.0.0.1:5000/api/v1/search?str=home&name=natgeotravel'

    );

 
    useEffect(() => {
    const fetchData = async () => {
      const result = await axios(url);
 
    setData(result.data);
  };

  fetchData();

  },[url]);
 
```
On définit ensuite les textbox correspondant au texte et à l'identifiant utilisateur cherchés, ainsi que le bouton pour lancer la recherche :
```javascript
  return (

    <Fragment>
      <input
        type="text"
        value={query}
        onChange={event => setQuery(event.target.value)}
      />
      <input
        type="texts"
        value={name}
        onChange={event => setName(event.target.value)}
      />
      <button type="button" onClick={() => setUrl(`https://photosearch-back.herokuapp.com/api/v1/search?str=${query}&name=${name}`)}>
        Search
      </button>
```
Et on renvoie enfin notre liste d'images.

```javascript

      <div className="card-list">
          {data.map((pic) => (
          <div className="card" key={data.id}>
            <img
              className="card--image"
              src={pic[1]}   
              alt={pic[4]}
              title={pic[2]}                      
              width="50%"
              height="50%"              
            ></img>

          </div>
        ))}
        </div>

     </Fragment>
  );
}
```

On peut maintenant lancer notre application react à partir du dossier racine de l'application :
```console
$yarn start
```
![img17](/assets/images/project_image_searcher/im17.png)

Notre application est maintenant disponible à l'adresse affichée. Elle affiche bien les images envoyées par notre API :

![img16](/assets/images/project_image_searcher/im16.png)




L'application est hébergée ici : <https://photosearch-app.netlify.app/>. Lors du lancement de la première requète, il est possible que la réponse soit assez longue, le temps du redémarrage du serveur. N'hésitez pas à la tester !


