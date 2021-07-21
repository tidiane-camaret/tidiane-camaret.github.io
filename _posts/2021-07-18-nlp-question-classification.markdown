---
layout: post
title:  "Traitement du langage :  Comprendre le sens d'une question "
date:   2021-07-18 11:20:25 +0100
categories: nlp python data_science
lang: fr
ref: question_classif
---


Lorsqu’on recherche une information dans de larges volumes de textes, la manière la plus répandue est l’utilisation de **mots clés**. Par exemple, si on cherche la date du début de la révolution industrielle, on peut chercher des termes susceptibles d’être présents dans des textes contenant la réponse cherchée : “révolution industrielle date” ou "révolution industrielle début”.

Mais certains moteurs de recherche sont capables d’interpréter des questions posées en langage naturel. Par exemple, si on tape dans la barre de recherche Google “Qui a écrit les misérables ?”, l’algorithme est en mesure de détecter que l’entité recherchée est un auteur, sans que le mot “auteur“ soit explicitement présent :

![img1](/assets/images/question_classif/im1.png)

Google utilise fréquemment WikiData, une base de données relationnelle où chaque entité est connectée à plusieurs autres en fonction de leurs rapports logiques. Connaître le type d’information recherchée au préalable facilite cette recherche.

Comment ces algorithmes arrivent-ils à saisir le sens d’une question ? On se propose ici de construire un programme avec le but suivant : Pour une question donnée, on veut trouver quelle est l’entité cherchée (un lieu ? une durée ? une distance ? une personne ?)  

# Une approche naïve : construire "à la main" des champs lexicaux 

La première idée qui pourrait nous venir en tête serait de définir des règles basées sur les champs lexicaux des catégories : par exemple, si une question contient une des déclinaisons des verbes “écrire”, “rédiger”,  elle a de grandes chances de porter sur un auteur. Comment définir ces champs lexicaux ?


On va ici utiliser une base de données contenant 5452 questions, la base ![TREC](https://search.r-project.org/CRAN/refmans/textdata/html/dataset_trec.html), pour Text REtrieval Conference. 

Chaque question, en anglais, est classée parmi 6 catégories :   
ABBR (Abbreviation)
DESC (Description and abstract concepts)
ENTY (Entities)
HUM (Human beings) 
LOC (Locations) 
NYM (Numeric values)

Et 50 sous-catégories, dont nous pouvons retrouver le détail ![ici](https://cogcomp.seas.upenn.edu/Data/QA/QC/definition.html)


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

Nous allons, à partir de ces champs lexicaux créés, construire un classificateur de questions. L'idée est simple : Pour chaque mot de la phrase, nous allons calculer sa probabilité d'appartenir à chaque catégorie. 

```python
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







# Partie 2 : Le RNN
