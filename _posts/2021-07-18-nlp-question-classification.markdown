---
layout: post
title:  "Traitement du langage :  Comprendre le sens d'une question "
date:   2021-07-18 11:20:25 +0100
categories: nlp python data_science
lang: fr
ref: question_classif
use_math: true
---


Lorsqu’on recherche une information dans de larges volumes de textes, la manière la plus répandue est l’utilisation de **mots clés**. Par exemple, si on cherche la date du début de la révolution industrielle, on peut chercher des termes susceptibles d’être présents dans des textes contenant la réponse cherchée : “révolution industrielle date” ou "révolution industrielle début”.

Mais certains moteurs de recherche sont capables d’interpréter des questions posées en langage naturel. Par exemple, si on tape dans la barre de recherche Google “Qui a écrit les misérables ?”, l’algorithme est en mesure de détecter que l’entité recherchée est un auteur, sans que le mot “auteur“ soit explicitement présent :

![img1](/assets/images/question_classif/im1.png)

Google utilise fréquemment WikiData, une base de données relationnelle où chaque entité est connectée à plusieurs autres en fonction de leurs rapports logiques. Connaître le type d’information recherchée au préalable facilite cette recherche.

Comment ces algorithmes arrivent-ils à saisir le sens d’une question ? On se propose ici de construire un programme avec le but suivant : Pour une question donnée, on veut trouver quelle est l’entité cherchée (un lieu ? une durée ? une distance ? une personne ?)  

# Une approche naïve : construire "à la main" des champs lexicaux 

La première idée qui pourrait nous venir en tête serait de définir des règles basées sur les champs lexicaux des catégories : par exemple, si une question contient une des déclinaisons des verbes “écrire”, “rédiger”,  elle a de grandes chances de porter sur un auteur. Comment généraliser cette idée et proposer un champ lexical pertient pour chaque catégorie ?


On peut pour commencer définir clairement les catégories qu'on va utiliser. Pour ça, on peut se baser sur des phrases déja écrites et classées. 


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

# Construction d'un classificateur


On va, à partir de ces champs lexicaux créés, construire un classificateur de questions. L'idée est simple : Pour chaque mot de la phrase, nous allons calculer sa probabilité d'appartenir à chaque catégorie. 

On va utiliser la méthode bayésienne :

Imaginons que notre base de données se composent uniquement de 3 questions, chacune dans une catégorie différente. 

--How many departments are there in France ?  **(count)**

--Who is the current french president ?     **(ind)**

--When did french people rebel against their king ?    **(date)**



Voici une 4e phrase : How many people live in France ?

Comment savoir à laquelle des 3 catégories elle appartient ?

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


On veut calculer la probabilité que notre phrase appartienne à une catégorie donnée, par exemple la catégorie **count**, sachant qu'elle contient les mots "How many people are in France". On va noter cette probabilité $P(count/"How\ many\ people\ are\ in\ France")$

le [théorème de bayes](https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_de_Bayes) nous permet d'écrire l'équation suivante : 

$P(count/"How\ many\ people\ are\ in\ France") = \frac{P("How\ many\ people\ are\ in\ France"/count) * P(count)}{P("How\ many\ people\ are\ in\ France")} $

P(count/"How many people are in France") = P("How many people are in France"/count) * P(count) / P("How many people are in France")

Puisque la phrase "How many people are in France" n'apparait pas dans notre jeu de données intial, on ne peut pas calculer 

On fait l'hypothèse que l'apparition d'un mot est un phénomène indépendant de celle des autres mots. On peut décomposer la probabilité comme suivant: 

Let us test some inline math $x$, $y$, $x_1$, $y_1$.

$P(count/"How\ many\ people\ are\ in\ France")$

P("How many people are in France"/**count**) = P("How"/**count**) * P("many"/**count**) * P("people"/**count**) * P("are"/**count**) * P("in"/**count**) * P("France"/**count**)

Regardons maintenant chacune de ces expressions. P("How"/**count**) exprime la probabilité de rencontrer le mot "How" dans une question de catégorie **count**. Or, ce mot apparait 1 fois dans la catégorie, qui compte en tout 7 mots : P("How"/**count**) = 1/7

Le mot "many" apparait également une fois sur 7 : P("many"/**count**) = 1/7

Le mot "people" n'apparait pas dans la catégorie **count**. On va associer à cette probabilité une valeur arbitrairement petite, mais non nulle, pour éviter que le produit tombe à zéro. P("many"/**count**) = 10^⁻3

et ainsi de suite. 

En résumé , 





```python
#On importe de la librairie sklearn des fonctions utiles au comptage des mots.

from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing,feature_selection, metric


dtf_train, dtf_test = model_selection.train_test_split(data, test_size=0.3)

y_train = dtf_train["Category2"].values


corpus = dtf_train["Questions"] #ensemble des textes des questions d'entrainement

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

Notre classificateur est bon dans la moitié des cas environs.


# Testing out latex

$$ \nabla_\boldsymbol{x} J(\boldsymbol{x}) $$