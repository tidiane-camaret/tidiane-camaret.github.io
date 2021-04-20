---
layout: post
title:  "analyse d'image : décrire une image "
date:   2021-04-18 11:20:25 +0100
categories: computer_vision react python data_science
lang: fr
ref: image_searcher
---

Ces dernieres années, plusieurs algorithmes d'analyse d'image, permettant afin d'extraire de l'information d'une image donnée, se sont popularisés. "Extraire de l'information" a ici un sens large : On peut vouloir extraire d'une image des informations basiques, telles que son taux de luminosité, ou sa netteté Mais des méthodes développées ces dernières années permettent de tirer d'une image des informations de plus haut niveau : détecter si l'image contient un visage, si elle contient du texte, ou tel ou tel objet ...

Un champ d'étude de l'analyse d'images, l'image captioning, se penche sur la capacité d'un algorithme à décrire le contenu d'une image, le plus souvent par une phrase plus ou moins longue.

J'ai voulu d'écrire une application mettant en oeuvre un de ces algorithmes, pour me permettre à la fois d'explorer en détail son fonctionnement, mais aussi de comprendre les étapes de construction d'une application : constitution d'une base de données, hébergement sur un serveur, conception d'une interface utilisateur ... 

J'ai décidé de le mettre en application sur une banque de photos bien connue sur internet : Instagram. 

Il semble qu'Instagram applique déja de lui mème, sur chaque image postée par un utilisateur, un algorithme de description d'image. 
Cependant, la description faite semble souvent assez vague :
En plus, il n'est pas possible d'effectuer une recherche sur la description générée.

Nous allons donc voir comment écrire une application permettant, sur un profil donné, de chercher les images correspondant le plus à une description cherchée.



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

