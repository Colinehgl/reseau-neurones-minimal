## Où en est le projet — bilan complet

### Objectif

Construire un **MLP (perceptron multicouche)** entièrement depuis zéro, en C, sans aucune bibliothèque externe — juste `stdlib`, `math.h` pour les fonctions élémentaires (`expf`, `tanhf`), et tes propres boucles pour tout le reste : algèbre linéaire, activations, forward, backward, entraînement. L'idée est de comprendre mécaniquement chaque rouage d'un réseau de neurones, sans la magie d'un framework.

### Choix structurants posés au départ

- **Type numérique** : `scalar_t = float` partout, plutôt que `double` — suffisant en précision pour du ML, deux fois moins gourmand en bande passante mémoire.
- **Représentation des matrices/vecteurs** : stockage à plat en row-major (`data[i*cols+j]`), pas de tableau de pointeurs — plus cache-friendly, allocation/libération en un seul bloc.
- **Gestion d'erreurs** : `assert()` partout dans les fonctions bas niveau — la plupart des erreurs possibles (dimensions incompatibles) sont des bugs de logique à corriger en développement, pas des cas à gérer proprement en production.
- **Structure de projet** : `src/` pour le code, `tests/` pour un fichier de test par module, `Makefile` avec une règle générique `test_<module>` qui compile et exécute chaque test indépendamment (`make test` les lance tous).
- **Séparation `Matrix`/`Vector`** : deux types distincts plutôt qu'un seul type matriciel générique (où un vecteur serait une matrice à une colonne) — plus explicite à l'usage, quitte à dupliquer certaines opérations (`matrix_hadamard`/`vector_hadamard`, `matrix_add`/`vector_add`).
- **Activations** : un enum `ActivationType` (`SIGMOID`, `TANH`, `RELU`) plutôt que des pointeurs de fonction — plus simple à débugger pour un premier projet en C, au prix d'un `switch` dans les fonctions de dispatch.

### Ce qui est terminé et testé

**`matrix.c` / `matrix.h`** — 25/25 tests. Le cœur de l'algèbre linéaire : `matrix_create`/`free`, `matrix_mul` (produit matriciel, ordre de boucles i-k-j pour la localité cache), `matrix_vec_mul`, `matrix_transpose`, `matrix_add`/`vector_add`, `matrix_hadamard`/`vector_hadamard`, `matrix_outer_product`, et les conversions `vector_to_matrix`/`matrix_to_vector`.

**`activation.c` / `activation.h`** — 29/29 tests. Sigmoid, tanh, ReLU et leurs dérivées, en versions scalaire, `Vector` et `Matrix`, plus les fonctions de dispatch `activation_apply`/`activation_apply_prime` qui font le lien avec l'enum `ActivationType`.

**`layer.c` / `layer.h`** — en cours de finalisation. Le type `Layer` (poids `W`, biais `b`, gradients `dW`/`db`, activation) et `LayerCache` (`z`, `a`, `delta`) sont posés. Les fonctions `layer_create`, `layer_free`, `layer_zero_grad`, `layer_forward`, `layer_backward_output`, `layer_backward_hidden` sont écrites ; le fichier de test vient d'être livré et plusieurs bugs restent à corriger avant validation complète (une erreur d'indexation dans `layer_init_weights`, et des fuites mémoire systématiques sur les résultats intermédiaires non libérés).

### Ce qu'il reste à faire

1. **Finir `layer.c`** — corriger les bugs identifiés, valider avec `test_layer.c`
2. **`loss.c` / `loss.h`** — MSE (et éventuellement cross-entropy pour de la classification), avec leurs dérivées ; c'est ce qui fournira le `loss_grad` dont `layer_backward_output` a besoin
3. **`mlp.c` / `mlp.h`** — assembler plusieurs `Layer` en un réseau complet : `mlp_create`, `mlp_forward` (enchaîne les couches, garde toutes les `LayerCache`), `mlp_backward` (parcourt les couches à l'envers, appelle `layer_backward_output` puis `layer_backward_hidden` en cascade)
4. **`train.c` / `train.h`** — la boucle d'epochs, la mise à jour SGD des poids à partir de `dW`/`db`, l'affichage de la loss
5. **`dataset.c` / `dataset.h`** — commencer par XOR en dur (4 exemples), prévoir la structure pour charger MNIST plus tard
6. **`main.c`** — assembler le tout : dataset, réseau, entraînement, affichage des résultats
7. **`tests/test_gradient_check.c`** — comparer le gradient analytique de la backprop à une approximation par différence finie, pour détecter tout bug de calcul de gradient qui aurait échappé aux tests unitaires

