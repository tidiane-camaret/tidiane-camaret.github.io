---
layout: post
title:  "Traitement du langage :  Comprendre le sens d'une question "
date:   2021-07-18 11:20:25 +0100
categories: nlp python data_science
lang: fr
ref: question_classif
use_math: true
---


Si on recherche une information précise dans de larges volumes de textes, une manière qui peut nous venir à l'esprit est d'utiliser des **mots clés** : Par exemple, si on cherche le nom de l'auteur **des Misérables**, on peut d'abord séléctionner un texte traitant de la littérature française en général, puis chercher à l'interieur de ce texte des termes susceptibles d’être à proximité de la réponse cherchée : **“Misérables auteur”** ou **"Misérables écrivain”**.

Mais certains moteurs de recherche sont capables d’interpréter des questions posées en langage naturel. Par exemple, si on tape dans la barre de recherche Google “Qui a écrit les misérables ?”, l’algorithme est en mesure de détecter que l’entité recherchée est un auteur, sans que le mot “auteur“ soit explicitement présent :

![img1](/assets/images/question_classif/im1.png)

<!--Google utilise fréquemment WikiData, une base de données relationnelle où chaque entité est connectée à plusieurs autres en fonction de leurs rapports logiques. Connaître le type d’information recherchée au préalable facilite cette recherche.-->

Comment ces algorithmes arrivent-ils à saisir le sens d’une question, et à savoir quel type d'information est recherchée ? On se propose ici de construire un programme avec le but suivant : Pour une question donnée, on veut trouver quelle est l’entité cherchée (un lieu ? une durée ? une distance ? une personne ?)  

# Une approche naïve : construire "à la main" des champs lexicaux 

La première idée qui pourrait nous venir en tête serait de définir des règles basées sur les champs lexicaux des mots de la question : par exemple, si celle-ci contient une des déclinaisons des verbes “écrire”, “rédiger”,  elle a de grandes chances de porter sur un auteur. Comment généraliser cette idée et proposer un champ lexical pertient pour chaque catégorie de question?


On peut, pour commencer, définir clairement les catégories dans lesquelles on va classer les questions. Pour ça, on peut se baser sur des phrases déja écrites et classées. 


On va ici utiliser un jeu de données contenant 5452 questions, la base [TREC](https://search.r-project.org/CRAN/refmans/textdata/html/dataset_trec.html), pour Text REtrieval Conference. Chaque question, en anglais, est classée parmi 6 catégories (**Abbreviation**, **Description and abstract concepts**, **Entities**, **Human beings**, **Locations et Numeric values**) et 50 sous-catégories, dont nous pouvons retrouver le détail [ici.](https://cogcomp.seas.upenn.edu/Data/QA/QC/definition.html)


Regardons d'abord à quoi ressemble notre jeu de données :

```python
#On extrait la table du fichier csv gràce à la librairie pandas
import pandas as pd
data = pd.read_csv('/media/tidiane/D:/Dev/NLP/data/Question_Classification_Dataset.csv', encoding='latin-1')
data
```

|      |                                         Questions |   Category0 | Category1 | Category2 |
|-----:|--------------------------------------------------:|------------:|----------:|----------:|
|   0  | How did serfdom develop in and then leave Russ... | DESCRIPTION | DESC      | manner    |
|   1  | What films featured the character Popeye Doyle ?  | ENTITY      | ENTY      | cremat    |
|  ... | ...                                               | ...         | ...       | ...       |
| 5447 | What 's the shape of a camel 's spine ?           | ENTITY      | ENTY      | other     |
| 5448 | What type of currency is used in China ?          | ENTITY      | ENTY      | currency  |
| 5449 | What is the temperature today ?                   | NUMERIC     | NUM       | temp      |
| 5450 | What is the temperature for cooking ?             | NUMERIC     | NUM       | temp      |
| 5451 | What currency is used in Australia ?              | ENTITY      | ENTY      | currency  |


Dans chaque catégorie, nous pouvons voir quels mots sont les plus utilisés : Ceux-ci sont les plus susceptibles de former un champ lexical cohérent.

```python
#On groupe les questions par catégorie 2 :
data_group = data.groupby(['Category2'])['Questions'].apply(lambda x: ' '.join(x)).reset_index()


#On transforme ensuite nos questions aggrégées en dictionnaire contenant la fréquence de chaque mot :

def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    return counts

data_group['count'] = data_group['Questions'].apply(word_count)

# Enfin, on affiche les 10 mots les plus fréquents de chaque catégorie :

for i in range(len(data_group)) :
    print(data_group.loc[i, "Category2"])
    count = data_group.loc[i, "count"]
    print(list(count.keys())[0:10], "\n")

```
![img2](/assets/images/question_classif/im2.png)

*Les 10 termes apparaissant le plus souvent dans chaque catégorie*

# Utilisation de l'inférence bayésienne


On va, à partir de ces champs lexicaux créés, construire un classificateur de questions, à partir de la [méthode bayésienne](https://fr.wikipedia.org/wiki/Inf%C3%A9rence_bay%C3%A9sienne). L'idée est simple : Nous allons calculer, pour chaque mot de la phrase, les probabilités d'appartenir à chaque catégorie, puis en déduire les probabilités pour la phrase entière.



Imaginons pour le moment que notre base de données se composent uniquement de 3 questions, chacune dans une catégorie différente : 

--How many departments are there in France ?  **(count)**

--Who is the current french president ?     **(ind)**

--When did french people rebel against their king ?    **(date)**

On peut en tirer un tableau listant la fréquence d'apparition des termes dans chaque catégorie : 

|             | count | ind | date |
|-------------|-------|-----|------|
| against     | 0     | 0   | 1    |
| are         | 1     | 0   | 0    |
| current     | 0     | 1   | 0    |
| departments | 1     | 0   | 0    |
| did         | 0     | 0   | 1    |
| France      | 1     | 0   | 0    |
| french      | 0     | 1   | 1    |
| How         | 1     | 0   | 0    |
| in          | 1     | 0   | 0    |
| is          | 0     | 1   | 0    |
| king        | 0     | 0   | 1    |
| many        | 1     | 0   | 0    |
| people      | 0     | 0   | 1    |
| president   | 0     | 1   | 0    |
| rebel       | 0     | 0   | 1    |
| the         | 0     | 1   | 0    |
| their       | 0     | 0   | 1    |
| there       | 1     | 0   | 0    |
| When        | 0     | 0   | 1    |
| Who         | 0     | 1   | 0    |


Voici une 4e phrase : **How many people live in France ?**

Pouvons nous savoir, à partir de nos trois phrases précédentes, à quelle catégorie elle appartient ?



Essayons par exemple de calculer la probabilité que notre phrase appartienne à la catégorie **count**, sachant qu'elle contient les mots "How many people are in France". On va noter cette probabilité $P(count/"How\ many\ people\ are\ in\ France")$

Le [théorème de bayes](https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_de_Bayes) nous permet d'écrire l'équation suivante : 

$P(count/"How\ many\ people\ are\ in\ France") = \frac{P("How\ many\ people\ are\ in\ France"/count) * P(count)}{P("How\ many\ people\ are\ in\ France")} $

<!--P(count/"How many people are in France") = P("How many people are in France"/count) * P(count) / P("How many people are in France")-->

Comment calculer la probabilité $P("How\ many\ people\ are\ in\ France"/count)$, c'est-à-dire la probabilité de tomber sur cette phrase exacte lorsqu'on tire une phrase dans la catégorie **count** au hasard ?

Notre jeu de données est bien trop réduit pour obtenir directement cette probabilité. On fait alors l'hypothèse simplificatrice que les apparitions individuelles des mots sont des événements indépendants entre eux. On peut alors décomposer la probabilité comme suivant: 

$P("How\ many\ people\ are\ in\ France"/count) = P("How"/count) * P("many"/count) * P("people"/count) * P("are"/count) * P("in"/count) * P("France"/count)$


Regardons maintenant chacune de ces expressions : $P("How"/**count**)$ exprime la probabilité de rencontrer le mot "How" dans une question de catégorie **count**. Or, ce mot apparait 1 fois dans la catégorie, qui compte en tout 7 mots (on peut se référer au tableau présenter plus haut) : $P("How"/count) = 1/7$

Le mot "many" apparait également une fois sur 7 : $P("many"/count) = 1/7$

Le mot "people" n'apparait pas dans la catégorie **count**. On va associer aux mots hors catégorie une **probabilité arbitrairement petite**, mais non nulle, pour éviter que le produit final ne tombe à zéro. $P("many"/count) = 10^{⁻3}$

Après avoir calculé la probabilité d'apparition de chaque mot, nous obtenons par leur produit le terme $P("How\ many\ people\ are\ in\ France"/count)$. Il est égal à $(1/7)⁵ * 10^{⁻3} \approx 5,94 * 10^{⁻8}$

De la même manière, $P("How\ many\ people\ are\ in\ France"/ind)= {10^{⁻3}}^6 = 10^{⁻18} $
 
Enfin, $P("How\ many\ people\ are\ in\ France"/date)= 1/7 * {10^{⁻3}}^5 \approx 1,42 * 10^{⁻16} $

On peut noter que les probabilités caclulées sont directement dépendantes du nombre de termes de la phrase présents dans chaque catégorie : 5 mots de la phrase sur 6 sont présents dans la catégorie **count**, contre 0 dans **ind** et 1 dans **date**.

On peut ensuite remarquer que les probabilités $P(count)$, $P(date)$ et $P(ind)$ sont toutes égales à $1/3$

Dans l'expression $\frac{P("How\ many\ people\ are\ in\ France"/ \textbf {categorie}) * P(\textbf {categorie})}{P("How\ many\ people\ are\ in\ France")} $, le seul terme non constant est celui que nous venons de calculer. C'est donc lui qui va determiner l'ordre de nos probabilités : $P(count/"How\ many\ people\ are\ in\ France") > P(date/"How\ many\ people\ are\ in\ France") > P(ind/"How\ many\ people\ are\ in\ France")$ 

Notre phrase initiale a donc, selon notre modèle, plus de chances d'appartenir à la catégorie **count**.

C'est cette idée d'inférence bayésienne que nous allons utiliser pour construire notre classificateur, en prenant cette fois-ci l'intégralité des phrases de notre base [TREC](https://search.r-project.org/CRAN/refmans/textdata/html/dataset_trec.html). On espère que la diversité des phrases présentes dans cette base contribuera à la robustesse de notre modèle. 

# Construction d'un classificateur bayésien

Nous allons diviser notre base de questions en deux. Une partie servira à entrainer notre modèle, et l'autre à évaluer ses performances. C'est une pratique courante qui évite d'évaluer un modèle sur une phrase sur lequel il a été entrainé, chose qui biaiserait nos résultats. 

```python
#On importe de la librairie sklearn des fonctions utiles au comptage des mots.

from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing,feature_selection, metric

#On sépare notre base en deux
dtf_train, dtf_test = model_selection.train_test_split(data, test_size=0.3)

y_train = dtf_train["Category2"].values


corpus = dtf_train["Questions"] #ensemble des textes des questions d'entrainement

#On crée notre table d'occurence des mots dans chaque catégorie
vectorizer = feature_extraction.text.CountVectorizer(max_features=10000)#, ngram_range=(1,1))
vectorizer.fit(corpus)

classifier = naive_bayes.MultinomialNB()
X_train = vectorizer.transform(corpus)

model = pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifier)])## train classifier
model["classifier"].fit(X_train, y_train)

X_test = dtf_test["Questions"].values
y_test = dtf_test["Category2"].values

predicted = model.predict(X_test)
predicted_prob = model.predict_proba(X_test)


metrics.accuracy_score(y_test, predicted) #0.4902200488997555
```

Notre classificateur a une précision d'environ 49%, ce qui est assez faible. Il faut cependant mettre ce résultat en perspective avec le fait que nous avons 50 catégories : le pur hasard nous donnerait un taux de réussite de 2% seulement.


# Une approche prenant en compte la structure du language : les word vectors

Notre modèle bayésien, s'il fait mieux que le pur hasard, possède une désavantage inhérent : Chaque occurence de mot est considérée comme indépendante des autres. 

Or, certains mots possèdent une proximité sémantique : les différentes conjuguaisons d'un même verbe, ou les noms de lieu, par exemple. Cette proximité n'est pas modélisée par notre approche, qui considère les mots comme des variables n'ayant aucun rapport entre elles.

L'utilisation de **word vectors** fait correspondre chaque **mot** à **un point dans un espace continu**, le plus souvent multidimensionnel. Ce procédé s'appelle **l'embedding**. 

Représenter les mots dans un tel espace permet de représenter la structure du language utilisé : les mots partageant un sens ou un attribut peuvent être placés proches les uns des autres. 

![img3](/assets/images/question_classif/im3.jpg)

Plusieurs méthodes existent pour produire des embeddings cohérents, et la plupart exploitent les co-occurences de mots dans des corpus de textes existants. L'hypothèse de base de ces méthodes est que des mots qui apparaissent dans des voisinages de mots similaires ont plus de chance de partager des attributs. Par exemple, les mots apparaisant après un pronom personnel ont de grandes chances d'être des verbes. 


Nous allons tenter de construire nos propres embeddings, en suivant la méthode **Continuous bag-of-words (CBOW)**, proposée par [Mikolov](https://arxiv.org/pdf/1301.3781.pdf). Cette méthode consiste à entrainer un **réseau de neurones** à prédire un mot à partir de ses mots voisins. 

Pour celà, nous allons exploiter notre base de données de questions. Notre réseau prendra en entrée un mot d'une question, et il devra correctement prédire un mot de son voisinage. 

Notre réseau a une couche cachée de taille égale à la dimension désirée de notre espace d'embeddings. A la fin de l'entrainement du réseau, on espère que les mots de sens similaires mèneront aux mêmes prédictions de mots voisins, et auront donc des activations de couche cachée similaires : c'est ces valeurs d'activations que nous considererons par la suite comme nos embeddings.


![img4](/assets/images/question_classif/im4.png)


On adapte d'abord notre jeu de données pour qu'il soit lisible par le réseau.