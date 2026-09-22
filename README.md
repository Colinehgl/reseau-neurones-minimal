## Où en est le projet — bilan complet

### Objectif

Construire un **MLP (perceptron multicouche)** entièrement depuis zéro, en C, sans aucune bibliothèque externe — juste `stdlib`, `math.h` pour les fonctions élémentaires (`expf`, `tanhf`). L'idée est de comprendre mécaniquement chaque rouage d'un réseau de neurones, sans la magie d'un framework.

### Choix structurants posés au départ

- **Type numérique** : `scalar_t = float` partout, plutôt que `double` — suffisant en précision pour du ML, deux fois moins gourmand en bande passante mémoire.
- **Représentation des matrices/vecteurs** : stockage à plat en row-major (`data[i*cols+j]`), pas de tableau de pointeurs — plus cache-friendly, allocation/libération en un seul bloc.
- **Gestion d'erreurs** : `assert()` partout dans les fonctions bas niveau — la plupart des erreurs possibles (dimensions incompatibles) sont des bugs de logique à corriger en développement, pas des cas à gérer proprement en production.
- **Structure de projet** : `src/` pour le code, `tests/` pour un fichier de test par module, `Makefile` avec une règle générique `test_<module>` qui compile et exécute chaque test indépendamment (`make test` les lance tous).
- **Séparation `Matrix`/`Vector`** : deux types distincts plutôt qu'un seul type matriciel générique (où un vecteur serait une matrice à une colonne) — plus explicite à l'usage, quitte à dupliquer certaines opérations (`matrix_hadamard`/`vector_hadamard`, `matrix_add`/`vector_add`).
- **Activations** : un enum `ActivationType` (`SIGMOID`, `TANH`, `RELU`) plutôt que des pointeurs de fonction — plus simple à débugger pour un premier projet en C, au prix d'un `switch` dans les fonctions de dispatch.
- **`LayerCache` construit via `layer_cache_create()`** plutôt qu'une déclaration brute — initialise `z`, `a`, `delta` à `NULL`, indispensable pour que `layer_forward` puisse libérer l'ancien contenu du cache sans crasher au tout premier appel. `layer_cache_free()` en est le symétrique pour tout libérer en fin d'usage.

### Ce qui est terminé et testé

**`matrix.c` / `matrix.h`** — 25/25 tests. Le cœur de l'algèbre linéaire : `matrix_create`/`free`, `matrix_mul` (produit matriciel, ordre de boucles i-k-j pour la localité cache), `matrix_vec_mul`, `matrix_transpose`, `matrix_add`/`vector_add`, `matrix_hadamard`/`vector_hadamard`, `matrix_outer_product`, et les conversions `vector_to_matrix`/`matrix_to_vector`.

**`activation.c` / `activation.h`** — 29/29 tests. Sigmoid, tanh, ReLU et leurs dérivées, en versions scalaire, `Vector` et `Matrix`, plus les fonctions de dispatch `activation_apply`/`activation_apply_prime` qui font le lien avec l'enum `ActivationType`.

**`layer.c` / `layer.h`** — terminé, tous les tests passent. Le type `Layer` (poids `W`, biais `b`, gradients `dW`/`db`, activation) et `LayerCache` (`z`, `a`, `delta`) sont en place. Fonctions : `layer_create`, `layer_free`, `layer_zero_grad`, `layer_init_weights` (Xavier), `layer_forward`, `layer_backward_output` (couche de sortie, point de départ de la backprop), `layer_backward_hidden` (couches cachées, propage le delta via `next_W`), `layer_cache_create`, `layer_cache_free`.

### Ce qu'il reste à faire

1. **`loss.c` / `loss.h`** — MSE (et éventuellement cross-entropy pour de la classification), avec leurs dérivées ; c'est ce qui fournira le `loss_grad` dont `layer_backward_output` a besoin
2. **`mlp.c` / `mlp.h`** — assembler plusieurs `Layer` en un réseau complet : `mlp_create`, `mlp_forward` (enchaîne les couches, garde toutes les `LayerCache`), `mlp_backward` (parcourt les couches à l'envers, appelle `layer_backward_output` puis `layer_backward_hidden` en cascade)
3. **`train.c` / `train.h`** — la boucle d'epochs, la mise à jour SGD des poids à partir de `dW`/`db`, l'affichage de la loss
4. **`dataset.c` / `dataset.h`** — commencer par XOR en dur (4 exemples), prévoir la structure pour charger MNIST plus tard
5. **`main.c`** — assembler le tout : dataset, réseau, entraînement, affichage des résultats
6. **`tests/test_gradient_check.c`** — comparer le gradient analytique de la backprop à une approximation par différence finie, pour détecter tout bug de calcul de gradient qui aurait échappé aux tests unitaires