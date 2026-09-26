## Où en est le projet — bilan complet

### Objectif

Construire un **MLP (perceptron multicouche)** entièrement depuis zéro, en C, sans aucune bibliothèque externe — juste `stdlib`, `math.h` pour les fonctions élémentaires (`expf`, `tanhf`, `logf`). L'idée est de comprendre mécaniquement chaque rouage d'un réseau de neurones, sans la magie d'un framework.

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

**`loss.c` / `loss.h`** — terminé, tous les tests passent. `mse_loss`/`mse_loss_prime` et `cross_entropy_loss`/`cross_entropy_loss_prime` — fournissent le `loss_grad` qui amorce la rétropropagation à la couche de sortie.

**`mlp.c` / `mlp.h`** — terminé, tous les tests passent. Le type `MLP` assemble plusieurs `Layer` et `LayerCache` en un réseau complet : `mlp_create`, `mlp_free`, `mlp_forward` (enchaîne les couches), `mlp_backward` (parcourt les couches à l'envers, propage `prev_a` et `delta` correctement d'une couche à l'autre), `mlp_zero_grad`.

**`train.c` / `train.h`** — terminé, tous les tests passent. `sgd_update` (descente de gradient : `W -= lr*dW`, `b -= lr*db`) et `train_epoch` (une passe complète sur un dataset : zero_grad, forward, loss_grad, backward, update, pour chaque exemple).

**`dataset.c` / `main.c`** — terminé. Dataset XOR codé en dur (4 exemples), assemblage complet dans `main.c` (réseau `{2,4,1}`, 10000 epochs, `lr=0.5`). **Premier entraînement de bout en bout réussi** : loss descendue de 0.25 à 0.00014, les 4 sorties XOR correctement apprises (0/1/1/0 attendus, sorties obtenues ≈ 0.011/0.989/0.989/0.012).

**`tests/test_gradient_check.c`** — terminé, tous les gradients cohérents. Compare le gradient analytique de `mlp_backward` (`dW`/`db`) à une approximation par différence finie centrée ($\epsilon=10^{-3}$) sur un réseau `{2,3,1}` : les 13 poids/biais vérifiés un par un concordent (tolérance relative 5%, avec un repli sur une tolérance absolue quand les deux valeurs sont déjà quasi nulles). Confirme que la backprop est mathématiquement correcte de bout en bout, indépendamment des tests unitaires précédents.

### Le projet est fonctionnellement complet

Toutes les briques prévues au départ sont écrites et validées : algèbre linéaire, activations, couche (forward/backward), loss, assemblage réseau, entraînement SGD, et vérification indépendante du gradient. Le MLP apprend XOR de bout en bout et sa backprop est confirmée correcte par gradient checking.

### Pistes d'extension, une fois le gradient checking fait

- Passer sur MNIST plutôt que XOR (nécessite un vrai chargeur de fichier binaire dans `dataset.c`, et sentir la limite de performance des boucles naïves sans BLAS)
- Mini-batch au lieu de SGD par exemple isolé
- Sauvegarde/chargement des poids sur disque