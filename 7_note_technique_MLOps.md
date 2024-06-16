# MLOps : note technique

Issue du *DevOps*, la démarche MLOps (*Machine Learning Operations*) sert à encadrer et automatiser les tâches nécessaires à l'élaboration d'un modèle d'apprentissage automatique, en vue de pouvoir sécuriser et répéter ces étapes et monitorer ses performances, dans un objectif d'amélioration continue.

## Démarche CI/CD

Le *continuous integration / continuous delivery* est au coeur du MLOps :
- l'**intégration continue** consiste à développer dans un environnement cloisonné, à versionner le code, automatiser la construction de l'environnement qui exécute ce code (build), à effectuer les tests unitaires et vérifier la qualité générale du code (Lint)
- le **déploiement continu** est le fait d'exécuter ce code au sein d'un environnement de production, à effectuer des tests de production, à surveiller et mesurer son exécution (artefacts, métriques, brèches de sécurité, etc.)

Les applications varient en fonction des projets mais la démarche est la même : chercher à **améliorer continuellement la qualité** du projet, que l'on peut résumer par le terme japonais "***kaizen***".

## MLOps en pratique

### Mise en place

Dans le cadre du projet actuel, voici une mise en place possible :
- mise en place d'un **conteneur de développement** (e.g. *Docker* ou machine virtuelle)
- mise en place du **versioning** (e.g. *GitHub*, *Framagit*, *GitLab*, ...)
- travail dans un **environnement de développement** dédié au projet (e.g. *Python venv*)
- mise en place d'outils de **contrôle de qualité du code** (e.g. *Pylint*)
- mise en place d'outils de **formatage du code** (e.g. *Black*, *Flake8*)
- **répertoriage des bibliothèques** utilisées dans un ou plusieurs fichiers (e.g. *Poetry* ou un simple fichier `requirements.txt` utilisable avec *Pip*)
- création de **tests unitaires** pour chaque fonction utilisée (e.g. *Pytest* ou *Unittest*)

### Côté ML

Les traitements d'apprentissage automatique ont aussi intérêt à être standardisés et automatisés :
- création de pipelines (avec *Scikit Learn Pipeline* par exemple)
  - un **pipeline de pré-traitement des données** pour uniformiser et répéter rapidement toutes les étapes de pré-traitement (e.g. supprimer les outliers, faire le feature engineering, etc.)
  - un **pipeline d'entraînement** des modèles** (e.g. pré-traiter les données d'entrée, les normaliser puis les réduire à 2 dimensions et enfin entraîner un modèle défini dessus), parfois regroupé avec le pré-traitement
  - un **pipeline d'inférence et / ou de production** (e.g. pré-traitement des entrées puis inférence sur tel modèle entraîné)
- **suivi des expérimentations** pour accélérer le choix d'un modèle ou simplement en garder une trace exploitable (e.g. *MLFlow*)

### Déploiement

Une étape importante est la mise en place d'**actions automatiques**, déclenchées par un évènement (e.g. *GitHub Actions*).  
Ici, on imagine qu'un *push* vers le répertoire de versioning déclenche certaines actions pour une mise en production :
- création du **conteneur**
- **installation** des bibliothèques requises
- **tests** unitaires
- **contrôle qualité** du code
- **formatage** du code
- exécution du code de **déploiement** de l'API si les actions précédentes ont été un succès


## *Model drift*

La démarche ne s'arrête pas une fois l'outil en production : il est aussi intéressant de suivre son évolution au fil du temps pour éviter le *model drift*.  

Le model drift se caractérise par une **dégradation des performances du modèle au fil du temps**.  
Typiquement sur l'exemple de ce projet, cela peut être des prédiction de tags en total décalage avec les questions utilisateurs.

On identifie ainsi deux dérives possibles :
- le ***data drift*** est l'évolution des **données d'entrée**. Les entraînements des modèles sont en décalage avec le terrain et doivent être refaits. Cela peut parfois être rapide, temporaire ou non, voire saisonnier (clustering de clients).
  > Typiquement, le modèle entraîné sur des données allant jusqu'en 2023 ne saura pas proposer un tag impliquant un langage informatique né en 2024.
- le ***concept drift*** est plus difficile à identifier et implique parfois de repenser le projet dans son ensemble. Il se produit lorsque le concept même du projet, à savoir sa **cible**, son **objectif**, évolue dans le temps.
  > Par exemple, il y a 20 ans, le concept de tag était différent et il n'y avait pas nécessité de disposer de 4 ou 5 tags pour une question : 1 ou 2 étaient clairs. Mais avec la muiltiplication des langages et des bibliothèques communautaires et des évolutions constantes, il est nécessaire pour bien cerner une question de proposer plusieurs tags généralistes et spécifiques.  
  > Cette fois, le réentraînement des données sur un même modèle ne suffit pas : il faut refaire des parties majeures de l'étude.

### Suivi des expérimentations

Afin de prévenir le model drift et / ou suivre l'utilisation d'un modèle, la démarche MLOps propose aussi plusieurs solutions :
- comme proposé plus haut, mise en place d'une bibliothèque de **suivi d'expérimentations** (e.g. *MLFlow*, facile à intégrer et avec un serveur de visualisation) qui conserve les données et artefacts à chaque lancement (pour l'inférence, cela peut prendre énormément de stockage, mieux vaut se limiter à des indicateurs très simples)
- **stockage de certains éléments** en direct lors des prédictions
  > temps d'inférence, question posée (titre + corps)
- **suivi de la performance des modèles** par des métriques adaptées
  > temps d'inférence (récupérable en direct), taux de couverture des tags utilisateurs (à tester a posteriori) et score Jaccard (idem, sur les questions et tags testés a posteriori)
- définir des **alertes en cas de dégradation** des performances
  > e-mail immédiat pour des performances trop basses, rapports hebdomadaires avec envoi des derniers résultats problématiques le cas échéant

Il existe plusieurs outils permettant de mettre en place un suivi des métriques avec alertes, avec chacun leurs avantages et inconvénients :
- **EvidentlyAI** est une solution commerciale avec une base open source, davantage tournée sur le LLM mais une part dédiée au ML généraliste.  
Elle permet d'utiliser des tableaux de bord, des tests, de la détection de dérive et de surveiller la qualité des données, bénéficie d'une **interface utilisateur conviviale** et d'une API.
- **Popmon** est une solution de suivi de population développée par ING, qui publie un package Python open source. Il se distingue par un profilage statistique du jeu de données et permet ainsi de **détecter efficacement un éventuel data drift** avec des rapports clairs et ergonomiques.
- **Promotheus** est open source et gratuit. C'est un outil de la CNCF (*Cloud Native Computing Foundation*) **simple, sécurisé et puissant** pour la collecte et l'analyse de métriques, pas nécessairement spécifique au ML. Il permet un stockage local indépendant et un système d'alertes à un "AlertManager" (email ou autre).
Sa gratuité, sa simplicité, sa sécurité et sa grande scalabilité en font un **choix premier**.

### Conclusion

Il serait ainsi intéressant d'utiliser **Popmon pour monitorer un data drift éventuel** tout en utilisant **Promotheus pour être alerté** d'une dérive du modèle visible via les métriques existantes.
