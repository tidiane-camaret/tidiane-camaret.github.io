---
layout: post
title:  "Description d'image :  décrire une image "
date:   2021-04-18 11:20:25 +0100
categories: computer_vision react python data_science
lang: fr
ref: image_searcher
---

Ces dernieres années, l'agumentation de la puissance de calcul des machines a permis la multiplication des méthodes permettant extraire de l'information d'une image donnée. "Extraire de l'information" a ici un sens large : On peut vouloir extraire d'une image des informations basiques, telles que son taux de luminosité, ou sa netteté. Mais certains algorithmes permettent de tirer d'une image des informations de plus haut niveau : détecter si l'image contient un visage, si elle contient du texte, ou tel ou tel objet ...

Un champ d'étude de l'analyse d'images, l'image captioning, se penche sur la capacité d'un algorithme à décrire le contenu d'une image par un texte plus ou moins long.

Le papier [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044), de Kelvin Xu et Al propose une architecture permettant d'associer à une image une phrase descriptive. Elle est constituée d'un CNN et d'un réseau récurrent (RNN) permettant de générer des phrases cohérentes et en rapport avec l'image. On va ici décrire brièvement le fonctionnement d'une telle architecture, et tenter de produire nos propres descriptions.



# Partie 1 : Le CNN

Une méthode traditionnelle de machine learning pour extraire les informations d'une image est l'utilisation d'un réseau de neurones, et plus particulièrement les réseau de neurones convolutif (CNN). Il applique des convolutions sur la surface de l'image, permettant d'y détecter des motifs de plus ou moins haut niveau, et d'associer ces motifs aux caratéristiques de tel ou tel objet. 

Lors de l'entrainement du réseau, celui-ci apprendra à la fois les caractéristiques des convolutions à appliquer, et à quel contenu associer ces motifs : visage, voiture, chat, etc ...



L'architecture traditionnelle d'un CNN comprend plusieurs étapes successives de convolution/normalisation, suivies d'une couche dite dense, où tous les neurones sont connectés : 
![img1](/assets/images/img_captioning/im1.png)
(source: [Kaggle.com](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist))

La taille de sortie de la couche dense correspond au nombre de classes que le réseau peut reconnaître. 

Des CNN déja entrainés sur de grandes bases de données d'images sont disponibles sur Internet. Parmi eux, [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), un CNN entrainé sur la base d'images [ImageNet](http://www.image-net.org/) (1.2 millions d'images de 1000 classes différentes), et publiée en 2012, reste encore aujourd'hui un référence de modèle pré-entrainé.


Celui-ci est disponible au téléchargement sur la [page Kaggle](https://www.kaggle.com/pytorch/alexnet) de la librarie python Pytorch. 



Ici, nous ne voulons pas associer une image à telle ou telle catégorie, mais bien en rédiger une phrase descriptive. Même en limitant la taille maximum de la phrase, le nombre de phrases différentes est beaucoup trop grand pour en associer chacune à un neurone de sortie de la couche finale.

Xu et Al montrent qu'il est toutefois possible de se servir de la capacité de représentation d'un CNN entrainé pour accomplir cette tache. Ils utilisent les couches dédiées à la convolution/normalisation, nécéssaires pour extraire les caractéristiques de l'image, mais décident d'extraire le vecteur généré à la fin de celui-ci, et de ne pas utliser la dernière couche dédiée à la classification. 

Chaque image est donc transformée en un vecteur de taille fixe. Ce vecteur contient un certain nombre d'informations sur l'image, même si il est ininterprétable tel quel. Xu et Al vont utiliser un autre réseau de neurones, un RNN, pour générer la phrase description à partir de ce vecteur : 

# Partie 2 : Le RNN

Les RNN sont un autre type d'architecture de réseau de neurones, particulièrement utilisées dans les taches impliquant un traitement séquentiel de l'information.

Comme dans la plupart des architectures, ces réseaux prennent un vecteur en entrée, et donnent un vecteur en résultat. Leur particularité est que le résultat dépend du vecteur d'entrée, mais aussi des vecteurs d'entrée recus précédemment.


Ceci fonctionne gràce au fait qu'à chaque étape du traitement, une partie des neurones du réseau conserve leur état pour l'étape suivante :

Cette capacité est particulièrement utile pour les taches nécéssitant une "mémorisation" des étapes précédentes pour traiter chaque étape, comme dans l'interprétation de signaux audio ou de texte. 




Xu et Al s'en servent pour la génération de la description d'image.

Le RNN prend comme état initial de sa couche cachée le vecteur représentatif de l'image généré par le CNN. 

On met en input du RNN un potentiel premier mot pour la description, par exemple le mot "Un". 

Le RNN va générer un autre mot, par exemple "homme", en modifiant au passage l'état de sa couche cachée.

C'est ce nouvel état qui va être utilisé lors de la génération du mot suivant.

En résumé, à chaque étape, le RNN utilisera comme entrée la sortie de l'étape précédente, comme l'état de sa couche cachée, l'état précédent.

![img2](/assets/images/img_captioning/im2.png)




# Partie 3 : Les données

On va entrainer notre architecture sur la base de données COCO. Cette base de données contient des centaines de milliers d'images annotées, est est disponible au téléchargement sur https://cocodataset.org/.
![img3](/assets/images/img_captioning/im3.png)


```python
from pycocotools.coco import COCO


coco = COCO(instances_train2014.json)

coco_caps = COCO(captions_train2014.json)

# get image ids 
ids = list(coco.anns.keys())
```

Regardons la première image de notre base :

```python
#import de quelques librairies pour l'affichage et la navigation
import matplotlib.pyplot as plt 
import skimage.io as io 
import numpy as np 
%matplotlib inline 

#extraction du premier id de la base
ann_id = ids[1]

#affichage de l'image et de son url fixe
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs( img_id )[0]
url = img['coco_url']
print(url)
I = io.imread(url)
plt.imshow(I)

#affichage des descriptions 
ann_ids = coco_caps.getAnnIds( img_id   )
anns = coco_caps.loadAnns( ann_ids )
coco_caps.showAnns(anns)

```
![img4](/assets/images/img_captioning/im4.png)

# Partie 4 : le traitement du texte

Pour permettre au RNN de traiter les phrases, on va les diviser en `tokens`, des unités d'informations que le réseau va traiter une à une. Chaque mot différent rencontré dans le texte de la base de donnée sera associé à un `token`, lui même indexé par un indice. Nous créeons également 3 `tokens` spécifiques, réservés à la compréhension du texte : 

**<start>** indiquera le début d'une description, **<end>** en indiquera la fin, et **<unk>** sera utilisé pour indiquer les éventuels non répertoriés. 



# Partie 4 : L'architechture