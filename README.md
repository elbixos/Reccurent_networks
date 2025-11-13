
<script type="text/javascript" async src="//cdn.bootcss.com/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

## Réseaux Récurrents

Ce cours a été rédigé, de mémoire, par Manuel Clergue, pour un projet maintenant aussi abandonné que les réseaux récurrents.

Néanmoins il me semble intéressant d'en garder une trace ici...

### Motivations

Jusqu’à présent, les données que nous fournissons à nos réseaux de neurones sont non ordonnées, c’est-à-dire que l’ordre de leur traitement n’est pas pertinent. Et d’ailleurs, lors de l’entraînement du réseau, il est fortement conseillé de les passer dans un ordre aléatoire et changé à chaque époque, pour justement éviter que l’apprentissage soit influencé par un ordre artificiellement créé. Après tout, une image de chat doit rester une image de chat, qu’elle soit précédée ou non par une image de chien dans notre base d’exemples.

Cependant, il est parfois, pour ne pas dire souvent, nécessaire d’apprendre non seulement à reconnaître un objet, mais également reconnaître le déplacement de cet objet (ici les termes reconnaître, objet et déplacement sont à prendre dans le sens le plus général). Prenons un exemple simple et parlant : imaginons une balle (ou tout autre objet à votre convenance) se déplaçant et que nous connaissions seulement la position de cette balle. Il peut être intéressant de pouvoir prédire la position future de la balle. Cette position future est déterminée par les lois de la physique, et dépend de la vitesse et de l’accélération de la balle (et quelques autres paramètres), données dont nous ne disposons pas. Pour pouvoir faire notre prédiction, il est nécessaire d’avoir les positions de la balle dans le passé, sa trajectoire. La position future de la balle peut être ainsi déterminée par ses positions, présentes et passées.

![positions](images/prernn1.svg)

Même si nous ne connaissons pas la physique de l’environnement dans lequel la balle se meut, on peut toujours s’en sortir, en analysant des trajectoires, c’est-à-dire des séquences de positions, et en apprenant les positions futures à partir des positions passées. C’est l’objectif de la modélisation de séquences, tâche pour laquelle les réseaux de neurones récurrents sont par construction très efficaces.

La notion importante est ici celle de séquence. Une séquence de données est une suite ordonnées de données. La nature des données dépend du problème que l’on cherche à résoudre : une suite de positions, comme pour notre exemple précédent, une suite d’images formant une vidéo, une suite de lettres, ou de mots, formant un texte, une suite de sons (ou de notes de musique), une suite de relevés de températures … Les principes que nous utiliserons seront les mêmes, seuls changeront les traitements préalables des données, et éventuellement les dimensions des entrées et des sorties du modèle.

Les applications de la modélisation de séquences sont nombreuses et rentrent dans ces trois classes :

- **prédiction** : déterminer la suite d’une séquence,
- **classification** : associer une catégorie à une séquence,
- **génération** : produire une séquence.

Là où les modèles précédents étaient limités à faire des correspondances un-à-un (une entrée, une sortie), les modèles de séquences peuvent faire des correspondances séquence-à-un (une séquence de données en entrée, une sortie). Par exemple, si on souhaite classer le type de musique à partir d’une partition, si on souhaite prédire la valeur future d’une séquence de données. On peut également souhaiter faire des correspondances un-à-séquence (une donnée d’entré, une séquence en sortie), comme par exemple avec les systèmes de génération de légendes d’images, ou même séquence-à-séquence, dont l’exemple le plus connu est la traduction automatique de texte.

![applications](images/prernn4.svg)

Une **séquence** est une *suite de données ordonnée dans le temps*. 

La **modélisation de séquence** est l'activité qui consiste à *chercher les relations temporelles qui existent entre les données*.

Les réseaux de neurones récurrents sont efficaces pour la modélisation de séquence. Il est néanmoins légitime de se demander si une nouvelle architecture de réseau est réellement nécessaire pour cette tâche. En effet, les réseaux de neurones classiques semblent suffisants. Nous pourrions considérer une séquence comme une donnée, qu’on fournirait en bloc.

Une caractéristique des réseaux profonds est leur robustesse, c'est à dire leur capacité à accepter tous types de données, y compris des séquences. Et avec les réseaux convolutifs, nous avons vu que nous pouvons apprendre efficacement des relations de proximité entre des composantes des entrées, comme avec les pixels d’une image. Les réseaux de neurones récurrents le font plus efficacement, mais nous verrons dans la première section de ce chapitre qu’un réseau profond classique peut effectivement traiter des séquences. Nous en verrons les limites, et c’est ce qui nous permettra d’introduire les réseaux récurrents dans la section suivante.

Cette nouvelle architecture nécessite une modification du processus d’apprentissage, et la rétro-propagation à travers le temps, présentée ensuite est une méthode d’apprentissage spécifique aux réseaux récurrents.

Enfin, je présenterai les réseaux **Long Short Time Memory** et les réseaux **Gated Recurrent Unit** qui sont des évolutions des réseaux récurrents.

### Réseaux profonds pour la modélisation de séquences

Comme nous l’avons vu en introduction, rien n’empêche, en principe, d’utiliser des réseaux profonds pour traiter des séquences. Il suffit de considérer une séquence comme un bloc qui sera fourni au réseau, et nous nous retrouvons avec une application courante des réseaux profonds.

Pour illustrer cela, reprenons notre exemple de la balle qui se déplace dans un plan. Pour simplifier, nous considérerons que la zone de déplacement de la balle est le carré unité dans le plan, et que la balle rebondit sur les parois du carré. Nous ne savons rien de la physique du système (gravité, plasticité, friction, …). Tout ce que nous avons ce sont des enregistrements de la position du centre de la balle mesurée au cours du temps à intervalle régulier.

Imaginons que nous souhaitions connaître la position de la balle au temps $$t$$ en connaissant sa position lors des $$n$$ pas de temps précédents. Vu comme cela, c’est une tâche de prédiction, mais nous pouvons très bien la ramener à une tâche de régression (la régression est processus par lequel nous essayons de modéliser une fonction à partir d’exemples) :

$$ \mathbf{x}^{t} = f(\mathbf{x}^{t-1},\mathbf{x}^{t-2},\ldots,\mathbf{x}^{t-n} ) $$

avec $$\mathbf{x}^{i}$$, le vecteur correspondant à la position de la balle au temps $$i$$.

Pour approximer cette fonction avec une réseau de neurones, nous pouvons découper dans nos enregistrements de trajectoires des blocs de $$n+1$$ positions. Les $$n$$ premières seront fournies en entrées du réseau, et la dernière jouera le rôle de la sortie désirée. Notre réseau en entrée prendra un vecteur de dimension $$n \times 2$$ ($$n$$ couples $$(x,y)$$) et fournira en sortie un vecteur de dimension $$2$$ (un couple $$(x,y)$$).

![prediction par NN standard](images/prernn2.svg)

La prédiction de trajectoire, vue comme la prédiction de la position future en fonction des n positions passées, peut être réalisée par un réseau de neurones classique.

Comme toujours avec un réseau profond classique, lors de l’apprentissage la sortie du réseau est comparée à la sortie désirée, ici la position suivant les $$n$$ positions en entrée du réseau, pour calculer l’erreur commise. Comme il ne s’agit pas de données discrètes, pour le calcul de l’erreur nous ne pouvons pas utiliser l’entropie croisée. La racine de la moyenne des carrées fera l’affaire.

La construction des exemples peut se faire de façon exhaustive : si nous disposons d’une trajectoire de $$N$$ points, on peut en extraire $$N-n-1$$ blocs de $$n+1$$ points. Ou bien ne prendre qu’un sous-ensemble de ces blocs. De même, parmi les blocs de la base d’apprentissage, certains serviront pour l’apprentissage, d’autres pour estimer l’erreur de généralisation. Que du classique !

Le choix de $$n$$ (l’historique de la séquence qu’on souhaite prendre en compte) dépend du problème, mais pour notre problème de balle, quelques pas de temps, entre 5 et 10, devraient suffire (après tout, pour estimer l’accélération à partir d’une trajectoire 3 pas de temps sont suffisants).

Les exemples (les blocs de positions) peuvent (doivent, en fait) être passés au réseau dans le désordre lors de l’apprentissage. L’ordre des positions est conservé à l’intérieur des blocs, et c’est à partir de cet ordre que l’apprentissage se fait. Mais ce sont des considérations qui échappent au réseau. La seule chose qu’il cherche à faire c’est d’approximer la fonction $$f$$.

Et sur cet exemple simple, il est fort à parier qu’il devrait assez bien s’en sortir. Pour corser la difficulté, nous pourrions non pas essayer de prédire la position à un pas de temps, mais à $$k$$ pas de temps après la séquence fournie en entrée. Le principe reste le même.

Une fois l’apprentissage réalisé, c’est-à-dire une fois que les paramètres (les poids des connections et les biais !) sont fixés, pour utiliser notre modèle, il suffit de lui fournir en entrée une séquence de $$n$$ positions de notre balle puis de récupérer en sortie la position future prédite.

Et si nous voulons non pas prédire la prochaine position de la balle, mais la trajectoire future ? Il suffit dans ce cas de prendre la sortie du réseau et de construire une nouvelle séquence en prenant la séquence initiale, en lui enlevant la première position, en lui ajoutant la sortie du modèle et en la réinjectant dans le modèle, pour obtenir la prochaine position de la trajectoire prédite. Et ainsi de suite, jusqu’à obtenir une trajectoire de la longueur désirée.


![Prediction à plusieurs pas de temps](images/prernn3.svg)

Bien évidemment, au fur et à mesure, les erreurs de prédiction vont avoir tendance à s’accumuler et à partir d’un certain nombre de pas de temps (**horizon de prédiction**) notre trajectoire prédite deviendra peu fiable, et ce d’autant plus vite que le système que nous essayons de modéliser est complexe et non linéaire.

L'**horizon de prédiction** d'un modèle correspond à la durée qu'il est capable de prédire correctement dans le futur.

Le même principe peut être utilisé pour faire de la classification. Là, plutôt que d’essayer de prédire une position future d’une trajectoire, nous pourrions essayer de déterminer un type de trajectoire : cercle, droite, parabole, … Notre base d’exemple serait construit à partir de morceaux de trajectoire, associés à son type. Et nous nous retrouvions dans un cas de classification classique, avec les mêmes outils et les mêmes techniques utilisés par exemple pour la classification d’image. Rien de différent, à part la construction de la base d’exemple.

Parce que vous commencez à être des connaisseurs des réseaux profonds, vous sentez bien qu’on rate peut-être quelque chose en procédant de la sorte. Lorsque nous analysons des images, nous avons vu qu’il n’était pas forcément nécessaire de chercher à déterminer d’un coup les relations entre tous les pixels, mais qu’il pouvait être intéressant de déterminer avec une première couche des patterns locaux (les relations entre des pixels voisins), puis dans une couche suivante les relations entre les patterns trouvés avec la couche précédente, et ainsi de suite. C’est ce qui avait conduit à la définition des réseaux convolutifs. Le même principe peut s’appliquer ici, en considérant non plus un voisinage spatial, mais temporel. Pour le réseau, peu importe que le vecteur représente une séquence temporelle plutôt qu’une image (aux dimensions près, bien sûr : une image est de dimension 2, une séquence temporelle est de dimension 1), nous pouvons utiliser le même principe.

Comme pour l’analyse d’image, en utilisant des réseaux convolutifs, nous pouvons espérer obtenir des modèles plus compacts (avec moins de paramètres, donc plus faciles à entraîner) pour des performances équivalentes. Et surtout, comme pour les images, les modèles convolutifs supportent plus facilement d’être utilisés avec des séquences de tailles différentes.

Nous avons vu l’utilisation d’un réseau profond (dense ou convolutif) pour la modélisation d’une séquence de points. Et cela semble bien fonctionner. Mais que ce passe t-il lorsque notre séquence est composée de données plus grandes, et que les relations pertinentes se font entre des éléments plus éloignés dans la séquence, que la dynamique que nous cherchons à modéliser est plus complexe ? Pour modéliser des séquences, nous voulons que nos modèles soient capables de :

1. manipuler des séquences de longueur variables,
2. prendre efficacement en compte l’ordre dans la séquence,
3. capturer les dépendances temporelles à long terme.

Un réseau dense permet de répondre aux critères 2 et 3. Il est cependant probable que cela implique, pour obtenir des résultats satisfaisants, que notre réseau soit plus volumineux (avec plus de couches, plus de neurones dans les couches, et donc plus de paramètres à apprendre), et que la convergence (l’annulation de l’erreur commise lors de l’apprentissage) se fasse plus lentement, voire ne se fasse pas du tout. Un réseau convolutif répondra lui au critère 1, plus difficilement au critère 2, et quasiment pas au critère 3, du fait de l’invariance translationnelle.

C’est pourquoi, il est apparu utile de définir une architecture de réseaux de neurones dédiées au traitement de séquences : les réseaux de neurones récurrents.

### 