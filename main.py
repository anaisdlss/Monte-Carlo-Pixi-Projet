"""
Projet : HP folding (lattice 2D) avec pull moves (méthode de l'article) + 
Monte-Carlo simple.
- On ne fait que des "pull moves" (corner / simple / propagé), car c'est la 
méthode la plus efficace selon l'article.
- La séquence d'acides aminés (AA) est convertie en séquence HP (H/P).
- Monte-Carlo avec Métropolis : propose un voisin (pull move), accepte/rejette, 
et mémorise le meilleur état.
- Visualisation : meilleure conformation, H en ronds bleus et P en ronds 
marron, centrés dans les cases de la grille.
"""

from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Circle
import random
import math
import matplotlib.pyplot as plt

# ----------------------
# Conversion AA -> HP
# ----------------------
HP_MAP = {
    "A": "H", "V": "H", "L": "H", "I": "H", "M": "H", "F": "H", "W": "H", "Y": "H",
    "R": "P", "N": "P", "D": "P", "C": "P", "Q": "P", "E": "P", "G": "P", "H": "P",
    "K": "P", "P": "P", "S": "P", "T": "P"
}


def conversion_hp(seq_aa: str) -> str:
    """Convertit une séquence d'acides aminés (ex: 'ACDE...') en chaîne HP (ex: 'HPPH...').

    Lève un KeyError si un code AA inconnu est présent.
    """
    return "".join(HP_MAP[aa] for aa in seq_aa)


# ----------------------
# Modèle de données
# ----------------------
class Residue:
    """Un résidu H/P placé sur la grille carrée (coordonnées entières)."""

    def __init__(self, typ: str, x: int, y: int):
        if typ not in ("H", "P"):
            raise ValueError("Residue.type doit être 'H' ou 'P'.")
        self.type = typ  # 'H' ou 'P'
        self.x = x       # abscisse (entier)
        self.y = y       # ordonnée (entier)


class Protein:
    """Chaîne HP en 2D (lattice carré), liaisons Manhattan (distance 1)."""

    def __init__(self, sequence_hp: str):
        """Construit une protéine HP initialement en ligne droite (x=0..n-1, y=0)."""
        if not sequence_hp or any(c not in ("H", "P") for c in sequence_hp):
            raise ValueError(
                "La séquence HP doit contenir uniquement 'H' ou 'P'.")
        self.sequence_hp = sequence_hp
        # Conformation initiale simple : tous les résidus alignés horizontalement
        self.residues = [Residue(t, i, 0) for i, t in enumerate(sequence_hp)]

    # ---------- utilitaires géométriques ----------
    def _is_adj(self, p, q) -> bool:
        """Vrai si p et q sont voisins 4-connexes (distance de Manhattan = 1)."""
        return abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1

    def _valid_chain(self, pos) -> bool:
        """Vrai si (i) aucune collision (positions toutes uniques)
        et (ii) liaisons de longueur 1 entre résidus consécutifs.
        Implémentation compacte avec 'set' (positions = tuples hashables).
        """
        # (i) unicité des positions
        if len(pos) != len(set(pos)):
            return False
        # (ii) continuité (chaque paire consécutive est adjacente)
        return all(self._is_adj(pos[i], pos[i+1]) for i in range(len(pos) - 1))

    # ---------- énergie HP ----------
    def energy(self) -> int:
        """Énergie HP standard : -1 par contact H-H non consécutif (sans double comptage)."""
        # Dictionnaire : (x,y) -> index du résidu (pour tester rapidement les voisins)
        coords_to_idx = {(r.x, r.y): i for i, r in enumerate(self.residues)}
        E = 0
        for i, r in enumerate(self.residues):
            if r.type != "H":
                continue  # seuls les H contribuent aux contacts
            # Les 4 voisins potentiels de la case (x,y)
            for nx, ny in ((r.x+1, r.y), (r.x-1, r.y), (r.x, r.y+1), (r.x, r.y-1)):
                j = coords_to_idx.get((nx, ny))  # index du voisin s'il existe
                # Contact H-H mais non consécutif le long de la chaîne
                if j is not None and self.residues[j].type == "H" and abs(i - j) > 1:
                    E -= 1
        # on divise par 2 car chaque contact a été vu deux fois (i<->j)
        return E // 2

    # ---------- copie ----------
    def copy(self) -> "Protein":
        """Copie indépendante de la protéine (nouveaux objets Residue)."""
        p = Protein(self.sequence_hp)
        p.residues = [Residue(r.type, r.x, r.y) for r in self.residues]
        return p

    # ---------- pull move (méthode de l'article) ----------
    def pull_move_inplace(self) -> bool:
        """Implémente un "pull move" fidèle à l'article :
        1) Tire UN résidu k uniformément au hasard.
        2) Énumère toutes les variantes de pull autour de k (2 orientations x 
        2 perpendiculaires), en gérant corner / pull simple / pull propagé.
        3) Conserve toutes les conformations valides (SAW + liaisons unité).
        4) S'il y en a au moins une, choisit UNE variante uniformément au 
        hasard et l'applique.

        Retourne True si une conformation a été appliquée, sinon False.
        """
        n = len(self.residues)
        if n < 3:
            return False  # trop court pour qu'un pull move soit possible

        # Instantané de la conformation (on s'y réfère pendant la propagation)
        old = [(r.x, r.y) for r in self.residues]
        # Occupation de la grille actuelle : (x,y) -> index
        occ = {p: i for i, p in enumerate(old)}

        # (article) Tirage d'un résidu k uniforme dans {0..n-1}
        k = random.randrange(n)

        candidates = []  # stocke toutes les nouvelles conformations valides générées

        # On teste les deux orientations : s=-1 (on tire vers N-ter), s=+1 (vers C-ter)
        for s in (-1, +1):
            iA = k                       # A = k (résidu tiré)
            # "pivot local" (opposé au sens de traction)
            downstream_idx = k - s
            upstream = k + s             # voisin "amont" (du côté où on tire)

            # Le pivot doit exister (évite bords inexploitables pour cette orientation)
            if not (0 <= downstream_idx < n):
                continue

            A = old[iA]
            downstream = old[downstream_idx]

            # Sécurité : A et le pivot doivent être adjacents (liaison)
            if not self._is_adj(A, downstream):
                continue

            # Vecteur A->pivot, puis deux perpendiculaires candidates
            dx, dy = downstream[0] - A[0], downstream[1] - A[1]
            for px, py in ((dy, -dx), (-dy, dx)):
                # case visée pour A
                L = (downstream[0] + px, downstream[1] + py)
                # l'autre coin du carré
                C = (A[0] + px, A[1] + py)

                # L doit être libre (sinon essaie l'autre perpendiculaire)
                if L in occ:
                    continue

                # ----- Corner move : C occupé par "upstream" -----
                if 0 <= upstream < n and occ.get(C) == upstream:
                    new = old[:]           # copie
                    new[iA] = L            # simple rotation A->L
                    if self._valid_chain(new):
                        candidates.append(new)
                    continue  # passe à l'autre perpendiculaire

                # Si C occupé par un autre OU pas d'upstream : move impossible ici
                if C in occ or not (0 <= upstream < n):
                    continue

                # ----- Pull simple / propagé (C libre & upstream existe) -----
                new = old[:]          # copie de travail
                new[iA] = L           # A -> L
                new[upstream] = C     # upstream -> C (on "accroche" la chaîne)

                # Test local d'adjacence à rétablir (formule unifiée par s)
                def check(pos, j): return self._is_adj(pos[j], pos[j - s])

                # Propagation : on "tire" en cascade tant que la liaison locale n'est pas recollée
                j = upstream + s            # point de départ de la propagation
                while 0 <= j < n and not check(new, j):
                    # source = ancienne case (j ± 2) selon le sens
                    src = j - 2 * s
                    if not (0 <= src < n):  # plus de source valide -> on abandonne cette branche
                        break
                    cand_pos = old[src]     # ancienne case à propager
                    # Évite la collision : si cand_pos est déjà occupée par un index ≠ j
                    if cand_pos in new and new.index(cand_pos) != j:
                        break
                    new[j] = cand_pos       # on place la case propagée
                    j += s                  # continue dans le même sens

                # Validation globale (SAW + continuité partout)
                if self._valid_chain(new):
                    candidates.append(new)

        # Aucune variante valable trouvée autour de k
        if not candidates:
            return False

        # Choix UNIFORME d'une variante parmi celles disponibles (évite les biais)
        chosen = random.choice(candidates)

        # Commit : applique la conformation retenue à l'objet courant
        for (x, y), r in zip(chosen, self.residues):
            r.x, r.y = x, y

        return True


# ----------------------
# Monte-Carlo simple (Métropolis)
# ----------------------
class MonteCarlo:
    """Monte-Carlo simple utilisant UNIQUEMENT des pull moves (façon article)."""

    def __init__(self, protein: Protein, temperature: float = 1.0, steps: int = 2000, seed: int | None = None):
        self.protein = protein              # état courant
        self.T = float(temperature)         # température Métropolis
        self.steps = int(steps)             # nombre d'itérations
        if seed is not None:                # option : graine pour reproductibilité
            random.seed(seed)
        self.curr_E = protein.energy()      # énergie actuelle
        self.best = protein.copy()          # meilleure protéine rencontrée
        self.best_E = self.curr_E           # meilleure énergie

    @staticmethod
    def _metropolis(dE: float, T: float) -> bool:
        """Critère d'acceptation Métropolis (accepte toujours si dE<=0)."""
        return dE <= 0 or random.random() < math.exp(-dE / T)

    def run(self):
        """Exécute MC et renvoie (dernier état, meilleur état, meilleure énergie)."""
        for _ in range(self.steps):
            # propose un voisin sur une copie
            cand = self.protein.copy()
            if not cand.pull_move_inplace():               # utilise EXCLUSIVEMENT des pull moves
                # aucun move dispo autour du k tiré -> itération nulle
                continue
            E_cand = cand.energy()                         # évalue l'énergie du voisin
            if self._metropolis(E_cand - self.curr_E, self.T):
                self.protein = cand                        # accepte : devient l'état courant
                self.curr_E = E_cand
                if self.curr_E < self.best_E:              # met à jour le meilleur
                    self.best_E = self.curr_E
                    self.best = cand.copy()
        return self.protein, self.best, self.best_E


# ----------------------
# Échelle de températures (géométrique)
# ----------------------
def geometric_temps(t_min=0.6, t_max=5.0, n=8):
    """
    Construit une échelle géométrique de températures croissantes.
    - t_min : température la plus basse (exploitation)
    - t_max : température la plus haute (exploration)
    - n     : nombre de répliques
    """
    if n < 2:
        return [float(t_min)]
    ratio = (t_max / t_min) ** (1.0 / (n - 1))           # raison géométrique
    # T_k = T_min * ratio^k
    return [t_min * (ratio ** k) for k in range(n)]


# ----------------------
# Replica-Exchange (Parallel Tempering) avec pull moves
# ----------------------
class ReplicaExchange:
    """
    Parallel Tempering avec *pull moves only* :
      - Plusieurs répliques de la même protéine à des températures différentes.
      - Chaque réplique fait du MC local (Métropolis) en n'utilisant QUE des pull moves.
      - À intervalles réguliers, on tente des échanges de *conformations* entre
        répliques voisines en température (critère d'échange RE).
      - On conserve le meilleur état global rencontré.
    """

    class _Replica:
        """Une réplique = (Protein, T, E courante, meilleur local)."""

        def __init__(self, protein, T):
            self.protein = protein              # état courant de cette réplique
            self.T = float(T)                   # température associée
            self.E = protein.energy()           # énergie courante
            self.best = protein.copy()          # meilleur état local
            self.best_E = self.E                # meilleure énergie locale

    def __init__(self, protein,
                 temps=None,           # liste de températures croissantes
                 sweeps=200,           # nb de propositions MC par réplique entre deux échanges
                 # nb de cycles d'échanges (chaque cycle fait 2 passes even/odd)
                 exchanges=200,
                 seed=None):
        """
        - protein   : instance Protein (état de départ cloné sur chaque réplique)
        - temps     : liste de T (si None -> échelle géométrique par défaut)
        - sweeps    : pas MC par réplique entre deux tentatives d'échange
        - exchanges : nb de cycles (sweep->échange paire / sweep->échange impaire)
        - seed      : graine RNG optionnelle
        """
        if seed is not None:
            random.seed(seed)

        # Échelle de températures
        self.temps = temps or geometric_temps()
        # Crée une réplique par température, en copiant la même protéine de départ
        self.replicas = [self._Replica(protein.copy(), T) for T in self.temps]
        self.sweeps = int(sweeps)
        self.exchanges = int(exchanges)

        # Meilleur global (toutes répliques confondues)
        self.best = self.replicas[0].best.copy()
        self.best_E = self.replicas[0].best_E
        self._refresh_global_best()  # initialise correctement

    # ---- Métropolis local ----
    @staticmethod
    def _metropolis(dE, T):
        """Acceptation locale (Métropolis)."""
        return dE <= 0 or random.random() < math.exp(-dE / T)

    def _mc_sweep(self, R):
        """
        Un sweep MC pour la réplique R :
          - propose 'sweeps' voisins par pull move,
          - accepte/rejette avec Métropolis à T = R.T,
          - met à jour meilleur local et global.
        """
        for _ in range(self.sweeps):
            cand = R.protein.copy()                 # propose sur une copie
            if not cand.pull_move_inplace():        # aucun pull move dispo sur ce tirage
                continue
            E_new = cand.energy()
            if self._metropolis(E_new - R.E, R.T):  # accepte ?
                R.protein = cand
                R.E = E_new
                # meilleur local
                if R.E < R.best_E:
                    R.best_E = R.E
                    R.best = R.protein.copy()
                # meilleur global
                if R.E < self.best_E:
                    self.best_E = R.E
                    self.best = R.protein.copy()

    # ---- Tentative d'échange entre répliques voisines ----
    def _attempt_swap(self, i, j):
        """
        Tente d'échanger les *conformations* entre répliques i et j.
        Critère RE : p = min(1, exp( (βi - βj) * (Ej - Ei) )).
        Si accepté, on échange (Protein, E) entre ces températures fixes.
        """
        Ri, Rj = self.replicas[i], self.replicas[j]
        beta_i, beta_j = 1.0 / Ri.T, 1.0 / Rj.T
        delta = (beta_i - beta_j) * (Rj.E - Ri.E)   # Δ = (βi-βj)*(Ej - Ei)
        if delta >= 0 or random.random() < math.exp(delta):
            # échange des états
            Ri.protein, Rj.protein = Rj.protein, Ri.protein
            Ri.E, Rj.E = Rj.E, Ri.E
            # mise à jour du meilleur global si nécessaire
            if Ri.E < self.best_E:
                self.best_E = Ri.E
                self.best = Ri.protein.copy()
            if Rj.E < self.best_E:
                self.best_E = Rj.E
                self.best = Rj.protein.copy()

    def _swap_pass(self, offset):
        """
        Une passe d'échanges pairés :
          - offset=0  : (0,1), (2,3), (4,5), ...
          - offset=1  : (1,2), (3,4), (5,6), ...
        """
        for i in range(offset, len(self.replicas) - 1, 2):
            self._attempt_swap(i, i + 1)

    def _refresh_global_best(self):
        """Recalcule le meilleur global à partir des répliques (utile au démarrage)."""
        for R in self.replicas:
            if R.E < self.best_E:
                self.best_E = R.E
                self.best = R.protein.copy()

    # ---- Boucle principale RE ----
    def run(self):
        """
        Exécute RE :
          Pour chaque cycle :
            - sweep MC sur chaque réplique, puis échanges (pairs),
            - sweep MC sur chaque réplique, puis échanges (impairs).
        Renvoie (meilleure_protéine_globale, meilleure_énergie_globale).
        """
        for _ in range(self.exchanges):
            # Passes even
            for R in self.replicas:
                self._mc_sweep(R)
            self._swap_pass(offset=0)
            # Passes odd
            for R in self.replicas:
                self._mc_sweep(R)
            self._swap_pass(offset=1)
        return self.best, self.best_E


# ----------------------
# Visualisation
# ----------------------

def plot_hp_lattice(xy, hp, E=None, title="Meilleure conformation",
                    r=0.40, pad=0.20, line_w=2.5):
    """
    xy  : liste de tuples (x, y) sur la grille (entiers, bords de cases)
    hp  : string/list de 'H'/'P' (même longueur que xy)
    E   : énergie affichée (optionnel)
    r   : rayon des disques en unités "data" (taille des points)
    pad : marge (en cases) ajoutée autour des limites pour éviter la coupe
    line_w : épaisseur du trait entre résidus
    """
    assert len(xy) == len(hp)

    # couleurs (garde ta palette ; change si tu préfères)
    colors = {'H': '#00cfe6', 'P': 'k'}

    # coordonnées entières (nœuds de la grille)
    xs, ys = zip(*xy)

    # centres des cases (points "dans" les cases, pas aux intersections)
    xc = [x + 0.5 for x in xs]
    yc = [y + 0.5 for y in ys]

    # bornes de la grille (bords des cases) + petite marge pour les disques
    xmin, xmax = min(xs) - pad, max(xs) + 1 + pad
    ymin, ymax = min(ys) - pad, max(ys) + 1 + pad

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    # trait entre résidus : noir, épais, sous les disques
    ax.plot(xc, yc, lw=line_w, color='black', zorder=1)

    # disques centrés dans les cases (légèrement plus petits pour voir le trait)
    for (cx, cy), t in zip(zip(xc, yc), hp):
        ax.add_patch(
            Circle((cx, cy), r, facecolor=colors[t],
                   edgecolor='black', linewidth=0.9, zorder=2)
        )

    # grille carrée liée aux données
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(math.floor(min(xs)), math.ceil(max(xs)) + 2))
    ax.set_yticks(range(math.floor(min(ys)), math.ceil(max(ys)) + 2))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True, which='major', linewidth=0.6, alpha=0.35)

    # alléger les axes si très long
    if (xmax - xmin) > 40:
        ax.xaxis.set_major_locator(MultipleLocator(5))
    if (ymax - ymin) > 20:
        ax.yaxis.set_major_locator(MultipleLocator(2))

    # bordures sobres
    for spine in ax.spines.values():
        spine.set_visible(False)

    # titre
    ax.set_title(f"{title} (E = {E})" if E is not None else title, pad=10)

    # légende compacte
    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=colors['H'],
               markeredgecolor='black', markersize=10, label='H'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=colors['P'],
               markeredgecolor='black', markersize=10, label='P'),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=True, framealpha=0.9)

    ax.tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    return fig, ax


# ----------------------
# Exemple d'utilisation (remplace aa_seq par ta séquence)
# ----------------------
if __name__ == "__main__":
    # Séquence exemple : chaîne B de l'insuline humaine
    aa_seq = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKE"
              "STLHLVLRLRGG")
    seq_hp = conversion_hp(aa_seq)

    # --- Baseline : MC simple (pull moves only) ---
    P0 = Protein(seq_hp)
    mc = MonteCarlo(P0, temperature=1.0, steps=10_000, seed=42)
    last_mc, best_mc, Ebest_mc = mc.run()
    print("[MC]  Meilleure énergie :", Ebest_mc)

    # --- Replica Exchange (Parallel Tempering) ---
    # Échelle de T (tu peux ajuster t_min/t_max/n selon la longueur de la chaîne)
    temps = geometric_temps(t_min=0.5, t_max=6.0, n=10)
    re = ReplicaExchange(Protein(seq_hp), temps=temps,
                         sweeps=500, exchanges=120, seed=42)
    best_re, Ebest_re = re.run()
    print("[RE]  Meilleure énergie :", Ebest_re)

    # --- Visualisation de la meilleure conformation (ici : RE) ---
    xy = [(r.x, r.y) for r in best_re.residues]
    hp = best_re.sequence_hp
    fig, ax = plot_hp_lattice(
        xy, hp, E=Ebest_re, title="RE - Meilleure conformation")
    plt.show()
