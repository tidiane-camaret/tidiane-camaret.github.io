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



L'application est hébergée ici : https://photosearch-app.netlify.app/. Elle est écrire en python pour le côté serveur (back) et en javascript, sous le framework ReactJs, pour le côté front.

# Partie 1 : Trouver des images : Comment fouiller Instagram

Tout d'abord, nous devons trouver un ensemble d'images sur lequel appliquer notre algorithme.
Nous voulons utiliser la plateforme Instagram, nous devons trouver un moyen de parcourir le site et d'en récupérer les photos. En effet, Instagram permet de consulter les images postées par un utilisateur donné, mais ne permet pas de télécharger en masse ces images, ni d'en récupérer leurs adresses. A première vue, la seule solution pour se constituer une banque d'images serait donc de parcourir les profils et de télécharger manuellement chaque image.

Heureusement, des programmes permettent de 'scraper' Instagram, c'est à dire de parcourir automatiquement le site afin de récupérer automatiquement les données qui nous intéressent.

Un de ces programmes, codé en python et nommé instagram-scraper, nous permet de télécharger l'ensemble des photos d'un profil donné. Nous allons essayer ceci sur un profil instagram comportant un grand nombre d'images diverses, un compte Instagram de la chaine National Geographic.

https://github.com/arc298/instagram-scraper

{% highlight python %}
pip install instagram-scraper
instagram-scraper natgeotravel  
{% endhighlight %}

Nos photos sont bien téléchargées. Le problème est que ce téléchargement prend un temps assez long. Nous allons récupérer uniquement les adresses url des images, car celles-ci nous suffirons pour accéder aux images plus tard.

{% highlight python %}
name = "natgeotravel"
!instagram-scraper $name --media-metadata --media-types none
{% endhighlight %}

Le temps d'execution est cettes fois-ci plus court.


# Partie 2 : Décrire une image : Comment analyser une image afin d'en tirer une description 

Maintenant que nous avons accès aux images, nous allons pouvoir assigner à chaqune d'entre elles une description. 
Une méthode traditionnelle de machine learning pour extraire les informations d'une image est d'utiliser des réseaux de neurones. Ces algorithmes prennent en entrée une image, ou plus particulièrement l'intensité de chaque pixel d'une image, et en ressortent 

Une architecture prisée en réseaux de neurones est le CNN

Malheureusement, plus d'un objet par image donc peu fiable

On utilise un générateur de descriptions. 

A la fin, nous avons une base de données comprenant l'url de chaque image, ainsi que 




# Partie 3 : Récupérer les meilleurs résultats : créer une API renvoyant les images les plus proches d'une descrption donnée



# Partie 3 : Afficher les résultats : créer une application interagissant avec l'API