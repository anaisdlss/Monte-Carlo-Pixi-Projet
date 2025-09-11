# Prérequis

Avant de commencer, assurez-vous d'avoir installé **Pixi** :  
[Guide d'installation Pixi](https://pixi.sh/dev/installation/)

Vous devez également avoir **git** installé sur votre machine.

---

# Installation et synchronisation

Clonez ce dépôt et déplacez-vous dans le dossier :

```bash
git clone https://github.com/anaisdlss/Monte-Carlo-Pixi-Projet.git Delassus_MonteCarlo
cd Delassus_MonteCarlo
```
Puis synchronisez l'environnement pixi:
```bash
pixi sync
```
---

# Utilisation

Dans un premier temps, lancez le script **hp_2d.py**:

```bash
pixi run python hp_2d.py
```
Suivez les instructions. L'execution devrait prendre quelques dizaines de seconde. Veuillez quitter la video générée pour poursuivre l'execution.
Les fichiers générés se trouvent dans le dossier **out2d/**.

Dans un second temps, lancez le script **hp_3d.py**:

```bash
pixi run python hp_3d.py
```
Suivez les instructions. L'execution devrait prendre quelques minutes. Veuillez quitter la video générée pour poursuivre l'execution.
Les fichiers générés se trouvent dans le dossier **out3d/**.


