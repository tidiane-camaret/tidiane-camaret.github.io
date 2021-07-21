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
compter la fréquence d’apparition des mots afin  d’estimer un champ lexical propre à la catégorie :













# Partie 2 : Le RNN

Les RNN sont un autre type d'architecture de réseau de neurones, particulièrement utilisées dans les taches impliquant un **traitement séquentiel** de l'information.

Comme dans la plupart des architectures, ces réseaux prennent un vecteur en entrée, et donnent un vecteur en résultat. Cependant, ils sont construits de manière à ce que le vecteur de résultat dépende du vecteur d'entrée, mais aussi des vecteurs d'entrée recus précédemment.

Ceci fonctionne gràce au fait qu'à chaque étape du traitement, **une partie des neurones du réseau conserve leur état de l'étape précédente**. Cette capacité est particulièrement utile pour les taches nécéssitant une "mémorisation" des étapes au fil du temps, comme le traitement de signaux audio ou de texte. 


Xu et Al s'en servent pour la génération de la description d'image :
Le RNN prend ici comme **état** le vecteur représentatif **I** généré à l'étape précédente. 

On met en **entrée** du RNN un potentiel premier mot pour la description, par exemple le mot **"Un"**. Le RNN génère un autre mot, par exemple **"homme"**, en **modifiant au passage son état**.

C'est ce **nouvel état** qui va être utilisé lors de la génération du mot suivant.


![img6fr](/assets/images/img_captioning/im6fr.png)

Le RNN génère donc séquentiellement une phrase, mot par mot. Intialement, sans entrainement du réseau, ces mots seront aléatoires et n'auront pas de rapport avec l'image initale. On va donc maintenant l'entrainer sur un grand nombre d'images :  


# Partie 3 : Les données

On va entrainer notre architecture sur la base de données **COCO**. Cette base de données contient **des centaines de milliers d'images annotées**, et est disponible au téléchargement sur https://cocodataset.org/.
![img3](/assets/images/img_captioning/im3.png)

On télécharge le fichier `annotations_trainval2014.zip`, contenant les informations de chaque image de la base à date de 2014, et on utilise la librairie `pycocotools`, qui contient des fonctions nous permettant de manipuler facilement ces informations.


```python
from pycocotools.coco import COCO

coco = COCO("instances_train2014.json") #instance de classe COCO, contenant des informations relatives à chaque image
coco_caps = COCO("captions_train2014.json") #instance contenant les descriptions (plusieurs par image)


ids = list(coco.anns.keys()) #liste des identifiants des images
```

On peut afficher une image de notre base, ainsi que ses descriptions associées :
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

Pour permettre au RNN de traiter les phrases, on va les diviser en **tokens**, des unités d'informations que le réseau va pouvoir traiter une à une. Chaque mot ou symbole rencontré dans le texte de la base sera associé à un token, lui même indexé par un indice dans un **dictionnaire**. Nous créeons également 3 tokens spécifiques, réservés à la compréhension du texte : 

`<start>` indiquera le début d'une description, `<end>` en indiquera la fin, et `<unk>` sera utilisé pour indiquer les éventuels mots non répertoriés dans le dictionnaire. 


```python
"""création de deux dictionnaires : word2idx et idx2word, pour passer rapidement 
d'un token à son indice correpondant"""
word2idx = {'<start>': 0,
 '<end>': 1,
 '<unk>': 2}
idx2word = {0: '<start>',
 1: '<end>',
 2: '<unk>'}

idx = 3

"""Création d'un compteur comprenant tous les tokens lus dans les descriptions """
coco = COCO("annotations/captions_train2014.json")
counter = Counter()
ids = coco.anns.keys()
for i, id in enumerate(ids):
    caption = str(coco.anns[id]['caption'])
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    counter.update(tokens)

    if i % 100000 == 0:
        print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

""" Ajout dans les dictionnaires word2idx et idx2word 
de tous les mots apparaissant plus de 8 fois"""
vocab_threshold = 8
words = [word for word, cnt in counter.items() if cnt >= vocab_threshold]

for i, word in enumerate(words):
    if not word in word2idx:
        word2idx[word] = idx
        idx2word[idx] = word
        idx += 1

len(word2idx) #taille du vocabulaire : 7072 tokens
```


# Partie 5 : Le traitement de l'image

Comme pour le texte des descriptions, l'image doit être mise sous forme vectorielle pour être traitée par le réseau. Nous la normalisons également pour la rendre plus facilement interprétable par le CNN utilisé.

On va créer notre architecture via les librairies **pytorch** et **torchvision** . Celles-ci intègrent des fonctions utiles pour le pré-traitement, comme `transforms` :

```python

from torchvision import transforms

transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # réduction à une taille 256 * 256
    transforms.ToTensor(),                           # conversion image->tenseur
    transforms.Normalize((0.485, 0.456, 0.406),      # normalisation
                         (0.229, 0.224, 0.225))])

```
# Partie 6 : Préparation de l'entrainement

On a maintenant toutes les informations nécessaires pour entrainer notre réseau.
**Pytorch** possède une classe pré-définie représentant les jeux de données, et il est possible créer une classe héritée de celle-ci pour les manipuler plus facilement. Un [tutoriel détaillé](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) disponible sur le site de Pytorch, mais en résumé, deux méthodes de la classe doivent être réecrites en fonction de notre jeu de données :


 - La méthode `__getitem__`, de manière à ce que `dataset[i]` renvoie le ième item,  
 - La méthode `__len__`, de manière à ce que `len(dataset)` renvoie la taille du jeu de données.  

Nous créons la classe héritée `CoCoDataset` :


```python
import torch.utils.data as data
class CoCoDataset(data.Dataset):
    
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.img_folder = img_folder
        if self.mode == 'train':            
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item['file_name'] for item in test_info['images']]
        
    def __getitem__(self, index):
        # obtention de l'image et de la description
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']

            # Conversion de l'image en tenseur
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)

            # Conversion de la description en tokens, puis en ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []

            caption.append(word2idx['<start>'])            
            caption.extend([word2idx[token] if token in word2idx else word2idx['<unk>'] for token in tokens])
            caption.append(word2idx['<end>'])
            
            caption = torch.Tensor(caption).long()

            return image, caption

        else:
            path = self.paths[index]

            # Conversion de l'image en tenseur
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            return orig_image, image

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        else:
            return len(self.paths)
```

On crée ensuite notre instance de classe `dataset` :
```python
from tqdm import tqdm

img_folder = 'cocoapi/images/train2014/'
annotations_file = 'cocoapi/annotations/captions_train2014.json'

os.chdir('/media/tidiane/D:/Dev/CV/Image_captioning')

dataset = CoCoDataset(transform=transform_train,
                          mode="train",
                          batch_size=200,
                          vocab_threshold=8,
                          vocab_file='./vocab.pkl',
                          start_word='<start>',
                          end_word='<end>',
                          unk_word='<unk>',
                          annotations_file=annotations_file,
                          vocab_from_file=False,
                          img_folder=img_folder)

num_workers=0
# Randomly sample a caption length, and sample indices with that length.
indices = dataset.get_train_indices()
# Create and assign a batch sampler to retrieve a batch with the sampled indices.
initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
# data loader for COCO dataset.
data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))


```
# Partie 7 : L'architechture

On va maintenant pouvoir constuire notre architecture :

Pour le CNN, on se base sur le réseau [resnet50](https://arxiv.org/abs/1512.03385), déja inclus dans la librairie `torchvision`. Comme indiqué précédemment, on retire la dernière couche du réseau, pour la remplacer par une couche de taille de sortie 256. Le vecteur **I** représentatif de chaque image aura donc cette image.

```python
import torch.nn as nn
import torchvision.models as models

embed_size = 256 

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True) #import de resnet50
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1] #supression de la dernière couche
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)        
        features = features.view(features.size(0), -1)        
        features = self.embed(features)        
        return features

```
Pour le RNN, on construit un réseau récurrent de taille d'entrée 256

```python

hidden_size = 100
vocab_size = len(word2idx)

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super( DecoderRNN , self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size 
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding( self.vocab_size , self.embed_size )
        self.lstm  = nn.LSTM(    input_size  =  self.embed_size , 
                             hidden_size = self.hidden_size,
                             num_layers  = self.num_layers ,
                             batch_first = True 
                             )
        self.fc = nn.Linear( self.hidden_size , self.vocab_size  )
        

    def init_hidden( self, batch_size ):
      return ( torch.zeros( self.num_layers , batch_size , self.hidden_size  ).to(device),
      torch.zeros( self.num_layers , batch_size , self.hidden_size  ).to(device) )
    
    def forward(self, features, captions):            
      captions = captions[:,:-1]      
      self.batch_size = features.shape[0]
      self.hidden = self.init_hidden( self.batch_size )
      embeds = self.word_embedding( captions )
      inputs = torch.cat( ( features.unsqueeze(dim=1) , embeds ) , dim =1  )      
      lstm_out , self.hidden = self.lstm(inputs , self.hidden)      
      outputs = self.fc( lstm_out )      
      return outputs

    def Predict(self, inputs, max_len=20):        
        final_output = []
        batch_size = inputs.shape[0]         
        hidden = self.init_hidden(batch_size) 
    
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden) 
            outputs = self.fc(lstm_out)  
            outputs = outputs.squeeze(1) 
            _, max_idx = torch.max(outputs, dim=1) 
            final_output.append(max_idx.cpu().numpy()[0].item())             
            if (max_idx == 1 or len(final_output) >=20 ):
                break
            
            inputs = self.word_embedding(max_idx) 
            inputs = inputs.unsqueeze(1)             
        return final_output  
```

# Partie 8 : L'entrainement du réseau 

On peut finalement entrainer notre réseau :

```python 
import math
import pickle

device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = EncoderCNN( embed_size )

decoder = DecoderRNN(  embed_size , hidden_size, vocab_size ,num_layers)

encoder.to(device)

images=  images.to(device) 
features = encoder(images)




batch_size = 200
num_layers =1 
num_epochs = 4
print_every = 1 #150
save_every = 1 

total_step = math.ceil( len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size   )



criterion = nn.CrossEntropyLoss()
lr = 0.001
all_params = list(decoder.parameters())  + list( encoder.embed.parameters() )
optimizer = torch.optim.Adam( params  = all_params , lr = lr  )



device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_save_path = 'models'
os.makedirs( model_save_path , exist_ok=True)

# Save the params needed to created the model :
decoder_input_params = {'embed_size' : embed_size , 
                'hidden_size' : hidden_size , 
                'num_layers' : num_layers,
                'lr' : lr ,
                'vocab_size' : vocab_size
                }

with open(  os.path.join(model_save_path , 'decoder_input_params_12_20_2019.pickle'), 'wb') as handle:
    pickle.dump(decoder_input_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

```

```python 

import sys 
for e in range(num_epochs):
  for step in range(total_step):
    indices = data_loader.dataset.get_train_indices()
    new_sampler = data.sampler.SubsetRandomSampler( indices )
    data_loader.batch_sampler.sampler = new_sampler    
    images,captions = next(iter(data_loader))    
    images , captions = images.to(device) , captions.to(device)
    encoder , decoder = encoder.to(device) , decoder.to(device)
    encoder.zero_grad()    
    decoder.zero_grad()
    features = encoder(images)
    output = decoder( features , captions )    
    loss = criterion( output.view(-1, vocab_size) , captions.view(-1) )
    loss.backward()
    optimizer.step()
    stat_vals = 'Epochs [%d/%d] Step [%d/%d] Loss [%.4f] ' %( e+1,num_epochs,step,total_step,loss.item() )
    if step % print_every == 0 :
      print(stat_vals)
      sys.stdout.flush()
    if e % save_every == 0:
      torch.save( encoder.state_dict() ,  os.path.join( model_save_path , 'encoderdata_{}.pkl'.format(e+1) ) )
      torch.save( decoder.state_dict() ,  os.path.join( model_save_path , 'decoderdata_{}.pkl'.format(e+1) ) )
```

# Partie 9 : Test de notre architecture

```python
cocoapi_loc=''

img_folder_test = os.path.join(cocoapi_loc, 'cocoapi/images/test2014/')
annotations_file_test = os.path.join(cocoapi_loc, 'cocoapi/annotations/image_info_test2014.json')

transform_test = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    #transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])


dataset_test = CoCoDataset(transform=transform_test,
                          mode='test',
                          batch_size = 1,
                          vocab_threshold=vocab_threshold,
                          vocab_file=' ',
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file_test,
                          vocab_from_file=False,
                          img_folder=img_folder_test)

data_loader_test = data.DataLoader(dataset=dataset_test,
                                      batch_size=dataset_test.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)


data_iter = iter(data_loader_test)

def get_sentences( original_img, all_predictions ):
    sentence = ' '
    plt.imshow(original_img.squeeze())
    return sentence.join([idx2word[idx] for idx in all_predictions[1:-1] ]  )

model_save_path = '/media/tidiane/D:/Dev/CV/Image_captioning/models/drive-download-20210428T070757Z-001'
os.makedirs( model_save_path , exist_ok=True)

device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(  os.path.join(model_save_path , 'decoder_input_params_12_20_2019.pickle'), 'rb') as handle:
    decoder_input_params = pickle.load(handle)

embed_size = decoder_input_params['embed_size']
hidden_size= decoder_input_params['hidden_size']
vocab_size = decoder_input_params['vocab_size']
num_layers = decoder_input_params['num_layers']

encoder = EncoderCNN( embed_size )
encoder.load_state_dict(   torch.load(  os.path.join( model_save_path , 'encoderdata_{}.pkl'.format(4) )   ,map_location=torch.device('cpu')) )

decoder = DecoderRNN( embed_size , hidden_size , vocab_size , num_layers )
decoder.load_state_dict( torch.load(   os.path.join( model_save_path , 'decoderdata_{}.pkl'.format(4) )  ,map_location=torch.device('cpu') ) )

encoder.to(device) 
decoder.to(device)
encoder.eval()
decoder.eval()
original_img , processed_img  = next( data_iter )

features  = encoder(processed_img.to(device)   ).unsqueeze(1)
final_output = decoder.Predict( features  , max_len=20)
get_sentences(original_img, final_output)
```

![img5](/assets/images/img_captioning/im5.png)
