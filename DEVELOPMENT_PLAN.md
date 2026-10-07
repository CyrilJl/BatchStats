# Plan de développement : statistiques incrémentales pour les bilans annuels

Date : 7 octobre 2026. Statut : lots 1 à 5 implémentés et vérifiés dans ce dépôt.
Cas d'usage initial : bilan annuel de qualité de l'air dans
`urban-aq`, avec calcul après-coup ou pendant la production des réanalyses.

API livrée : fusion de `BatchNanSum` / `BatchNanMean`, `BatchTopK` /
`BatchNanTopK`, méthodes `rank` et `quantile`, utilitaire `required_k`, état
versionné `to_state` / `from_state` et checkpoints NPZ `save` / `load`.
Les rangs indisponibles sont masqués pour préserver les dtypes ; `BatchTopK`
refuse les NaN. Guide et mesures dans `docs/source/extreme_statistics.rst`,
benchmark reproductible dans `benchmarks/extremes.py`. Les sections suivantes
conservent les objectifs et critères du plan initial. Les extensions reportées
et l'intégration urban-aq restent hors périmètre.

## Objectif et périmètre

Fournir des accumulateurs NumPy génériques, fusionnables et sauvegardables pour
traiter une série par lots sans charger toute sa dimension temporelle en mémoire.
Les mêmes accumulateurs doivent servir à lire des archives et à recevoir les
résultats d'une simulation au fil de leur production.

BatchStats porte les algorithmes numériques. Le calendrier, les seuils de qualité
de l'air, la complétude réglementaire, les unités, la provenance et la politique
de conservation restent dans urban-aq. Ce plan ne définit pas de règles juridiques.

Priorités :

1. Compléter la fusion des sommes et moyennes avec NaN.
2. Ajouter la sélection exacte des plus grandes ou plus petites valeurs.
3. Exposer les rangs et quantiles extrêmes exacts lorsque l'état retenu suffit.
4. Permettre la sauvegarde et la restauration des accumulateurs.
5. Documenter la lecture par blocs et mesurer les performances.

Les quantiles arbitraires approchés et les adaptateurs xarray/NetCDF sont des
extensions distinctes, pas des prérequis à cette première livraison.

## État du package vérifié

- Le cœur dépend uniquement de NumPy et propose `update_batch`, `__call__` et,
  pour plusieurs classes, la fusion par `+`.
- Les tableaux multidimensionnels et la réduction selon `axis` existent déjà.
  Un bloc `(time, y, x)` peut donc être réduit selon `axis=0`.
- `BatchNanSum` conserve une somme et un effectif valide par position.
- `BatchNanMean` repose sur `BatchNanSum`.
- Ces deux classes n'ont actuellement pas de fusion par `+`, contrairement à
  certaines autres classes, notamment les extrema avec NaN.
- Le package ne propose pas de sélection top-k ou de quantiles.

Avant modification, relire les classes de base et les tests : les conventions
de forme, de NaN et de dtype ne doivent pas être déduites des seuls noms de classes.

## Fondement : rangs et quantiles extrêmes

### Rang fixe

La k-ième plus grande valeur se calcule exactement en conservant les k plus
grandes valeurs rencontrées, avec leur multiplicité. La fusion de deux états
consiste à sélectionner les k plus grandes valeurs de leur union.

La mémoire persistante est en O(k × nombre de positions non réduites). Le même
principe s'applique aux plus petites valeurs. Il ne faut pas conserver une liste
Python ou un objet par cellule : les états doivent être des tableaux vectorisés.

### Quantile fixe

Pour la convention linéaire de NumPy, sur N valeurs valides triées en ordre
croissant, la position du quantile q dans [0, 1] est :

```text
h = (N - 1) * q
i = floor(h)
j = ceil(h)
Q(q) = x[i] + (h - i) * (x[j] - x[i])
```

Il s'agit d'une interpolation entre valeurs classées, pas d'une interpolation
temporelle. Aux positions entières, retourner directement la valeur de rang.

Pour une sélection des plus grandes valeurs, il faut retenir au moins :

```text
k_required(N, q) = N - floor((N - 1) * q)
```

Exemples avec cette convention :

| Série complète | Quantile | Valeurs supérieures nécessaires |
| --- | --- | --- |
| 365 moyennes journalières | P99 | 5 |
| 366 moyennes journalières | P99 | 5 |
| 8 760 moyennes horaires | P99,8 | 19 |
| 8 784 moyennes horaires | P99,8 | 19 |

Pour 365 valeurs, le P99 combine la cinquième plus grande valeur avec un poids
de 0,64 et la quatrième avec un poids de 0,36.

Un rang fixe demande une capacité fixe. Pour un quantile fixe, la capacité
nécessaire croît avec N. Une durée maximale connue permet de la dimensionner
avant le premier lot. Une médiane exacte ne devient pas un calcul à petite
mémoire grâce à cette méthode.

Avec des NaN, N est propre à chaque position. Un état suffisamment dimensionné
pour l'effectif maximal doit permettre le calcul avec les effectifs réellement
valides. Si les rangs nécessaires ont été éliminés, lever une erreur explicite :
ne jamais retourner un quantile silencieusement approché.

## Lot 1 — Fusion des statistiques avec NaN

Ajouter `__add__` à `BatchNanSum` et `BatchNanMean`, dans le style des accumulateurs
existants. Fusionner les sommes et les effectifs, pas les moyennes non pondérées.

Contrat :

- vérifier les axes et les formes des dimensions non réduites ;
- définir la promotion des dtypes et documenter le risque de débordement entier ;
- traiter les états non initialisés et les positions entièrement manquantes ;
- conserver le comportement public actuel sur une position sans valeur valide ;
- ne modifier aucun opérande et ne partager aucun tableau mutable avec le résultat ;
- documenter que l'ordre des sommes flottantes peut modifier les derniers bits.

Tests : lots inégaux, NaN spatialement variables, états vides, axes négatifs et
multiples selon les conventions existantes, incompatibilités et absence d'alias.
Comparer à NumPy sur la concaténation des lots et aux mises à jour séquentielles.

Critère de fin : somme et moyenne avec NaN sont utilisables en réduction parallèle,
avec des tolérances numériques justifiées plutôt qu'une promesse d'identité binaire.

## Lot 2 — Sélection exacte des valeurs extrêmes

API indicative, à stabiliser après vérification des conventions du package :

```python
state = BatchTopK(k=19, axis=0, largest=True)
state.update_batch(block)
merged = state + other
values = merged()
```

Prévoir une variante `BatchNanTopK` cohérente avec la famille `BatchNan*`, qui ignore
les NaN indépendamment à chaque position. Définir explicitement le comportement
de `BatchTopK` en présence de NaN ; ne pas hériter par accident d'un filtrage qui
supprimerait une heure entière pour toutes les cellules.

Décisions à inscrire dans la documentation de l'API :

- `k` entier strictement positif ; `largest=False` sélectionne les minima ;
- sortie classée selon le rang, avec un axe de rang clairement documenté ;
- forme fixe de cet axe et traitement explicite des rangs non disponibles ;
- effectif valide par position, y compris si moins de k valeurs ont été reçues ;
- multiplicité conservée : deux valeurs égales occupent deux rangs ;
- sémantique des valeurs infinies, distincte de celle des NaN ;
- compatibilité requise pour fusionner : capacité, direction, axes, formes ;
- aucune conversion float32 imposée et aucun arrondi des valeurs sélectionnées.

Pour la première version, une fusion de capacités différentes peut être refusée.
Les indices et dates des extrema sont hors périmètre initial.

Implémentation : sélection par partition plutôt que tri complet de l'historique ;
sélection locale du lot avant fusion avec l'état lorsque cela réduit les temporaires.
Mesurer la mémoire temporaire réelle, pas seulement celle de l'état persistant.

Tests : comparaison au tri NumPy, k=1, k supérieur à l'effectif, doublons, NaN,
infinis, lots vides, axes, formes incompatibles et entrées non contiguës. Vérifier
que les valeurs sélectionnées sont identiques quels que soient le découpage en
lots et l'arbre de fusion, et que les entrées ne sont pas modifiées.

## Lot 3 — Rangs et quantiles exacts à partir de la sélection

Exposer une lecture de rang explicite, avec des rangs commençant à 1, et une
lecture de quantile dont q est compris dans [0, 1]. La forme finale de l'API
(méthodes de l'accumulateur ou fonctions dédiées) sera arrêtée au lot 2.

La première version peut se limiter à `method="linear"`. Toute autre méthode
doit être refusée tant qu'elle n'est pas implémentée et testée. Une convention
supplémentaire, telle que `inverted_cdf`, fera l'objet de tests propres.

Le calcul doit :

- utiliser l'effectif valide de chaque position ;
- traiter N=0, N=1 et les positions de quantile entières ;
- vérifier que tous les rangs requis sont présents avant de calculer ;
- refuser une requête incompatible avec la capacité conservée ;
- définir une politique explicite pour les positions vides, sans les confondre
  avec une capacité insuffisante ;
- comparer le résultat à `numpy.quantile` ou `numpy.nanquantile` avec la même méthode.

Ajouter un utilitaire de dimensionnement à partir de q, d'un effectif maximal et
du côté conservé. Une éventuelle façade acceptant `max_samples` devra vérifier le
dépassement de cette borne ; augmenter k après avoir perdu des valeurs ne répare
pas l'historique.

Critère de fin : les exemples annuels ci-dessus et les séries incomplètes passent
les tests ; une médiane non calculable depuis un petit top-k est refusée clairement.

## Lot 4 — Sauvegarde et restauration

Définir un contrat d'état explicite, par exemple `to_state()` / `from_state()`,
contenant le type de statistique, une version de schéma, les paramètres, les formes,
les dtypes, les effectifs et les tableaux numériques nécessaires.

Commencer par les accumulateurs utilisés ici : sommes et moyennes avec NaN,
sélection de valeurs extrêmes. Documenter précisément les classes couvertes.

- Copier les tableaux pour éviter les alias avec l'accumulateur vivant.
- Valider paramètres, formes et versions à la restauration.
- Permettre un stockage JSON pour les métadonnées et NPZ pour les tableaux,
  sans dépendre de pickle ; lecture des tableaux avec `allow_pickle=False`.
- Laisser au consommateur les écritures atomiques, les checksums et l'identification
  des journées déjà traitées.

Tests : aller-retour avant et après initialisation, poursuite du calcul après
restauration, fusion d'états restaurés, état corrompu et version inconnue.

Une fusion n'est pas idempotente : fusionner deux fois une journée double sa
contribution. BatchStats ne peut pas détecter un chevauchement temporel sans
métadonnées métier. Ce contrôle appartient à urban-aq.

## Lot 5 — Lecture par blocs et exemples

Documenter d'abord un exemple sans dépendance supplémentaire : un itérateur
fournit des tableaux NumPy, puis appelle `update_batch`.

Pour NetCDF, le consommateur doit :

1. Lire une tuile spatiale et un bloc temporel bornés.
2. Décoder les valeurs manquantes et les tableaux masqués explicitement.
3. Conserver la correspondance des coordonnées et l'ordre des dimensions.
4. Alimenter l'accumulateur associé à cette tuile.
5. Finaliser ou sauvegarder l'état avant de passer à une autre tuile.

Ne jamais alimenter le même état avec des tuiles de positions différentes comme
s'il s'agissait de nouveaux instants. Les dimensions de sortie doivent représenter
les mêmes positions à chaque mise à jour.

Une extension optionnelle xarray pourra ensuite itérer selon une dimension nommée,
préserver les coordonnées et gérer les masques. Elle devra rester indépendante
du cœur NumPy et éviter toute matérialisation de l'année entière. Décider de son
ajout après un prototype dans urban-aq et une mesure des entrées-sorties.

## Intégration attendue dans urban-aq — hors de ce dépôt

- Un moteur commun pour le calcul après-coup et pendant la production.
- Des résumés journaliers remplaçables, identifiés par leur configuration et leurs
  entrées, puis une réduction annuelle reproductible.
- Des compteurs de dépassements horaires, des effectifs, et les rangs horaires
  nécessaires : les seules moyennes journalières ne suffisent pas.
- Une conservation optionnelle des moyennes journalières pour recalculer librement
  les quantiles journaliers ; une conservation horaire pour les demandes horaires
  arbitraires formulées ultérieurement.
- Une convention de précision commune : la quantification à l'écriture des NetCDF
  ne doit pas faire diverger les seuils entre calcul direct et relecture.
- Des contrôles de calendriers, couverture, doublons, scénarios, grilles et masques.
- Aucune suppression automatique des archives dans BatchStats.

Les quantiles annuels ne sont ni la moyenne ni le quantile des quantiles journaliers
ou mensuels. Fusionner les sélections et les effectifs, puis calculer le quantile.

## Validation et performance

Utiliser les tests existants (`test_merge.py`, `test_nanstats.py`, `test_axis.py`,
`test_edge_cases.py`) et ajouter un fichier dédié à la sélection et aux quantiles.
Conserver les cas mémoire importants dans des benchmarks distincts des tests CI.

Mesurer :

- temps et pic mémoire selon la taille des lots, des tuiles et k ;
- float32 et float64, données complètes et masquées ;
- sélection incrémentale face au tri et à la partition NumPy sur série complète ;
- coût de fusion et de sauvegarde ;
- résultat séquentiel face à plusieurs arbres de réduction.

Ordre de grandeur : un état de 19 valeurs float32 sur 5,2 millions de cellules
occupe environ 395 Mo par polluant, hors effectifs et temporaires. Le traitement
par tuiles reste nécessaire même avec une sélection courte.

Exécuter les vérifications du dépôt après chaque lot concerné : pytest, Ruff,
construction de la documentation ; puis construction et vérification du package
avant une livraison. Les ajouts doivent préserver la compatibilité Python et NumPy
déclarée dans `pyproject.toml`.

## Livraisons proposées

| Livraison | Contenu | Condition de validation |
| --- | --- | --- |
| A | Fusion de `BatchNanSum` et `BatchNanMean` | Équivalence NumPy, formes et NaN testés |
| B | Sélection top-k et rangs | Exactitude et fusion indépendantes du découpage |
| C | Quantiles extrêmes et dimensionnement | Convention explicite, capacité contrôlée |
| D | États sauvegardables | Reprise et fusion après restauration vérifiées |
| E | Exemples par blocs et benchmarks | Mémoire et temps mesurés sur un cas représentatif |

Chaque livraison ajoute sa documentation et ses tests. Les noms d'API sont à
stabiliser avant leur première publication. Aucun changement de version, commit
ou publication n'est effectué par la rédaction de ce plan.

## Extensions reportées

- Quantiles approchés génériques, avec contrat d'erreur documenté et comparaison
  des approches avant de choisir un algorithme.
- Quantiles pondérés.
- Dates et indices des valeurs extrêmes.
- Fenêtres glissantes et agrégations calendaires.
- Adaptateur xarray optionnel, si le prototype consommateur justifie sa généralisation.

Ces extensions ne doivent pas retarder les rangs et quantiles extrêmes exacts,
qui couvrent le besoin initial avec un contrat plus simple à vérifier.
