# MLOps : note technique

Issue du *DevOps*, la démarche MLOps (*Machine Learning Operating System*) sert à encadrer et automatiser les tâches nécessaires à l'élaboration d'un modèle d'apprentissage automatique, en vue de pouvoir sécuriser et répéter ces étapes et monitorer ses performances, dans un objectif d'amélioration continue.

## Démarche CI/CD

Le *continuous integration / continuous delivery* est au coeur du MLOps :
- l'**intégration continue** consiste à développer dans un environnement cloisonné, à versionner le code, automatiser la construction de l'environnement qui exécute ce code (build), à effectuer les tests unitaires et vérifier la qualité générale du code (Lint)
- le **déploiement continu** est le fait d'exécuter ce code au sein d'un environnement de production, à effectuer des tests de production, à surveiller et mesurer son exécution (artefacts, métriques, brèches de sécurité, etc.)

Les applications varient en fonction des projets mais la démarche est la même : chercher à **améliorer continuellement la qualité** du projet, que l'on peut résumer par le terme japonais "***kaizen***".

## En pratique

Dans le cadre du projet actuel, voici une mise en oeuvre MLOps possible :
- mise en place d'un **conteneur de développement** (e.g. *Docker* ou machine virtuelle)
- mise en place du **versioning** (e.g. *GitHub*, *Framagit*, *GitLab*, ...)
- travail dans un **environnement de développement** dédié au projet (e.g. *Python venv*)
- mise en place d'outils de **contrôle de qualité du code** (e.g. *Pylint*)
- mise en place d'outils de **formatage du code** (e.g. *Black*, *Flake8*)
- **répertoriage des bibliothèques** utilisées dans un ou plusieurs fichiers (e.g. *Poetry* ou un simple fichier `requirements.txt` utilisable avec *Pip*)
- création de **tests unitaires** pour chaque fonction utilisée (e.g. *Pytest* ou *Unittest*)
- création de pipelines (avec *Scikit Learn Pipeline* par exemple)
  - un **pipeline de pré-traitement des données** pour uniformiser et répéter rapidement toutes les étapes de pré-traitement (e.g. supprimer les outliers, faire le feature engineering, etc.)
  - un **pipeline d'entraînement** des modèles** (e.g. pré-traiter les données d'entrée, les normaliser puis les réduire à 2 dimensions et enfin entraîner un modèle défini dessus), parfois regroupé avec le pré-traitement
  - un **pipeline d'inférence et / ou de production** (e.g. pré-traitement des entrées puis inférence sur tel modèle entraîné)

Une étape importante est la mise en place d'**actions automatiques**, déclenchées par un évènement (e.g. *GitHub Actions*).  
Ici, on imagine qu'un *push* vers le répertoire de versioning déclenche certaines actions pour une mise en production :
- création du **conteneur**
- **installation** des bibliothèques requises
- **tests** unitaires
- **contrôle qualité** du code
- **formatage** du code
- exécution du code de **déploiement** de l'API



<!-- 🚧🚧🚧🚧🚧🚧 -->

Une fois en production, 

- **suivi de la performance des modèles** par des 
- 
- 



- GH Actions à chaque push
- MLFlow
- Présenter la conception concrète du système de suivi de la performance adapté au projet : les indicateurs et mesures à mettre en oeuvre, les types d’alerte préconisées (il n’est pas demandé de le développer)
- Suivi → surveillance “model drift” & “data drift” (définir rapidement) : voir evidentlyAI, Promotheus, ou Popmon
  - benchmark avec avantages / inconv des solut° (+ inclure prix)
- système de stockage d’événements relatifs aux prédictions réalisées par l’API et une gestion d’alerte en cas de dégradation significative de la performance.
- Présenter comment utiliser les outils envisagés pour mettre en oeuvre le système de suivi
