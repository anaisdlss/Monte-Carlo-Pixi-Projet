"""Anaïs DELASSUS - M2BI | HP folding 3D (pull moves) + MC + Replica-Exchange

- Grille cubique 3D (coordonnées entières)
- Énergie HP : -1 par contact H-H non consécutif (compté une fois)
- Mouvements: pull moves en 3D (corner / simple / propagé)
- MC Metropolis (kB=1)
- Replica Exchange (Parallel Tempering) avec 'sweeps' = N tentatives
- Vidéo MP4/GIF : structure 3D (gauche) + énergie (droite)
- PNG meilleure conformation

Dépendances : matplotlib
Pour MP4 : ffmpeg installé (sinon GIF fallback)
"""

import math
import random
import sys
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter, PillowWriter

# nécessaire pour proj='3d' (import side-effect)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# MP4 lisible partout (QuickTime-friendly)
QT_ARGS = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]

MASTER_SEED = 100

# ------------------ HP : conversion AA -> HP ------------------

HP_MAP = {
    "A": "H",
    "V": "H",
    "L": "H",
    "I": "H",
    "M": "H",
    "F": "H",
    "W": "H",
    "Y": "H",
    "R": "P",
    "N": "P",
    "D": "P",
    "C": "P",
    "Q": "P",
    "E": "P",
    "G": "P",
    "H": "P",
    "K": "P",
    "P": "P",
    "S": "P",
    "T": "P",
}

COLORS = {"H": "#00cfe6", "P": "#8c5a2b"}


def conversion_hp(seq_aa):
    """Convertit une séquence d'acides aminés (AA) en chaîne HP (H/P)."""
    return "".join(HP_MAP[aa] for aa in seq_aa)


# ------------------ Outils géométriques (3D) ------------------


def l1_adjacent(p, q):
    """Vrai si p et q sont voisins sur la grille 3D 
    (distance de Manhattan = 1)."""
    return abs(p[0] - q[0]) + abs(p[1] - q[1]) + abs(p[2] - q[2]) == 1


def add(p, q):
    """Somme de vecteurs 3D entiers."""
    return (p[0] + q[0], p[1] + q[1], p[2] + q[2])


def sub(p, q):
    """Différence de vecteurs 3D entiers (p - q)."""
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def unit_axes_3d():
    """Les 6 axes unitaires de la grille cubique 3D."""
    return [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
            (0, 0, -1)]


def perpendicular_dirs(v):
    """Renvoie les 4 directions perpendiculaires au vecteur axial v 
    (±x/±y/±z)."""
    vx, vy, vz = v
    if vx != 0:  # v // x
        return [(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    if vy != 0:  # v // y
        return [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]
    # vz
    return [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]


# ------------------ Modèle HP 3D (pull moves) -----------------


class Residue:
    """Un résidu H/P placé sur la grille 3D."""

    def __init__(self, typ, coord):
        if typ not in ("H", "P"):
            raise ValueError("Residue.type doit être 'H' ou 'P'")
        self.type = typ
        if len(coord) != 3:
            raise ValueError("Coordonnée 3D attendue (x,y,z).")
        self.coord = (int(coord[0]), int(coord[1]), int(coord[2]))


class Protein:
    """
    Chaîne HP sur réseau 3D (grille cubique). Coordonnées entières.
    État initial : ligne sur l'axe x (x=0..n-1, y=0, z=0).
    """

    def __init__(self, sequence_hp):
        if (not sequence_hp) or any(c not in ("H", "P") for c in sequence_hp):
            raise ValueError("Séquence HP invalide (uniquement 'H' et 'P').")
        self.sequence_hp = sequence_hp
        self.residues = [Residue(t, (i, 0, 0)) for i, t in
                         enumerate(sequence_hp)]

    def _valid_chain(self, pos):
        """Pas de collisions + adjacence unité entre voisins de la chaîne."""
        if len(pos) != len(set(pos)):
            return False
        for i in range(len(pos) - 1):
            if not l1_adjacent(pos[i], pos[i + 1]):
                return False
        return True

    def energy(self):
        """
        Énergie HP : -1 par contact H-H non consécutif (compté une seule fois).
        Voisinage = 6 voisins 3D (±x, ±y, ±z).
        """
        occ = {r.coord: i for i, r in enumerate(self.residues)}
        E = 0
        for i, r in enumerate(self.residues):
            if r.type != "H":
                continue
            for ax in unit_axes_3d():
                npos = add(r.coord, ax)
                j = occ.get(npos)
                if j is not None and self.residues[j].type == "H" and \
                        abs(i - j) > 1:
                    E -= 1
        return E // 2  # chaque contact vu deux fois

    def copy(self):
        p = Protein(self.sequence_hp)
        p.residues = [Residue(r.type, r.coord) for r in self.residues]
        return p

    def pull_move_inplace(self, rng):
        """
        Pull move 3D (corner / simple / propagé).
          1) Tire un index k au hasard (résidu A).
          2) Prend le voisin 'downtream' D (k±1) si adjacent 
          (arête de la chaîne).
          3) Choisit une direction perpendiculaire pdir (4 possibilités en 3D).
          4) Tente: corner (si l’upstream est déjà au coin C), 
          sinon simple/propagé.
        Applique UNE candidate valide au hasard. Renvoie True si appliqué.
        """
        n = len(self.residues)
        if n < 3:
            return False

        old = [r.coord for r in self.residues]
        occ = {pos: i for i, pos in enumerate(old)}
        k = rng.randrange(n)
        candidates = []

        for s in (-1, +1):  # -1: vers N-ter ; +1: vers C-ter
            iA = k
            downstream = k - s
            upstream = k + s
            if not (0 <= downstream < n):
                continue
            A = old[iA]
            D = old[downstream]
            if not l1_adjacent(A, D):
                continue

            v = sub(D, A)  # direction de la liaison (axiale)
            perps = perpendicular_dirs(v)  # 4 directions perpendiculaires

            for pdir in perps:
                L = add(D, pdir)  # nouvelle case pour A
                C = add(A, pdir)  # "coin"
                if L in occ:
                    continue

                # --- corner : l’upstream est déjà au coin C ---
                if 0 <= upstream < n and occ.get(C) == upstream:
                    new = old[:]
                    new[iA] = L
                    if self._valid_chain(new):
                        candidates.append(new)
                    continue

                # --- simple/propagé ---
                if C in occ or not (0 <= upstream < n):
                    continue
                new = old[:]
                new[iA] = L
                new[upstream] = C

                def ok(pos, j):
                    return l1_adjacent(pos[j], pos[j - s])

                j = upstream + s
                while 0 <= j < n and not ok(new, j):
                    src = j - 2 * s
                    if not (0 <= src < n):
                        break
                    cand_pos = old[src]
                    # collision ?
                    if cand_pos in new and new.index(cand_pos) != j:
                        break
                    new[j] = cand_pos
                    j += s

                if self._valid_chain(new):
                    candidates.append(new)

        if not candidates:
            return False
        chosen = rng.choice(candidates)
        for c, r in zip(chosen, self.residues):
            r.coord = c
        return True


# ------------------ MC Metropolis (pull moves) ----------------


class MonteCarlo:
    """Monte Carlo Metropolis sur modèle HP 3D, avec pull moves."""

    def __init__(self, protein, temperature=1.0, steps=2000, seed=MASTER_SEED):
        self.P = protein
        self.T = float(temperature)
        self.steps = int(steps)
        self.rng = random.Random(seed)
        self.E = self.P.energy()
        self.best = self.P.copy()
        self.best_E = self.E

    def run(self):
        """Effectue 'steps' tentatives ; met à jour l’état si accepté 
        (Metropolis)."""
        for _ in range(self.steps):
            cand = self.P.copy()
            if not cand.pull_move_inplace(self.rng):
                continue
            E_new = cand.energy()
            dE = E_new - self.E
            if dE <= 0 or self.rng.random() < math.exp(-dE / self.T):
                self.P = cand
                self.E = E_new
                if self.E < self.best_E:
                    self.best_E = self.E
                    self.best = self.P.copy()
        return self.P, self.best, self.best_E


# --------------- Ladder géométrique de températures -----------


def geometric_temps(t_min=0.5, t_max=6.0, n=10):
    """n températures géométriquement espacées entre t_min et t_max."""
    if n < 2:
        return [float(t_min)]
    r = (t_max / t_min) ** (1.0 / (n - 1))
    return [t_min * (r**k) for k in range(n)]


# ------------------ Replica Exchange (Parallel Tempering) -----


class ReplicaExchange:
    """
    Parallel Tempering (3D) :
      - M répliques à T différentes
      - alternance : (sweeps) -> échanges voisins -> (sweeps) -> échanges 
      décalés
      - IMPORTANT : 1 sweep = N tentatives de move (N = longueur de chaîne)
    """

    class _Rep:
        def __init__(self, protein, T, seed=MASTER_SEED):
            self.rng = random.Random(seed)
            self.P = protein.copy()
            self.T = float(T)
            self.E = self.P.energy()
            self.best = self.P.copy()
            self.best_E = self.E

    def __init__(self, protein, temps=None, sweeps=200, exchanges=60,
                 seed=MASTER_SEED):
        self.temps = temps or geometric_temps()
        self.reps = [self._Rep(protein, T, seed + i) for i, T in
                     enumerate(self.temps)]
        self.sweeps = int(sweeps)
        self.exchanges = int(exchanges)
        self.best = self.reps[0].best.copy()
        self.best_E = self.reps[0].best_E

    def _one_move(self, R):
        cand = R.P.copy()
        if not cand.pull_move_inplace(R.rng):
            return
        E2 = cand.energy()
        dE = E2 - R.E
        if dE <= 0 or R.rng.random() < math.exp(-dE / R.T):
            R.P, R.E = cand, E2

    def _sweep(self, R):
        """1 sweep = N tentatives de move."""
        N = len(R.P.residues)
        for _ in range(N):
            self._one_move(R)
        # best locaux + global
        if R.E < R.best_E:
            R.best_E = R.E
            R.best = R.P.copy()
        if R.E < self.best_E:
            self.best_E = R.E
            self.best = R.P.copy()

    def _try_swap(self, i, j, rng):
        """Tentative d’échange entre répliques i et j (critère d’échange 
        standard)."""
        Ri, Rj = self.reps[i], self.reps[j]
        beta_i, beta_j = 1.0 / Ri.T, 1.0 / Rj.T
        delta = (beta_i - beta_j) * (Rj.E - Ri.E)
        if delta >= 0 or rng.random() < math.exp(delta):
            Ri.P, Rj.P = Rj.P, Ri.P
            Ri.E, Rj.E = Rj.E, Ri.E

    def run(self, seed=MASTER_SEED):
        rng = random.Random(seed)
        for _ in range(self.exchanges):
            # passe 0-1, 2-3, ...
            for R in self.reps:
                for _ in range(self.sweeps):
                    self._sweep(R)
            for i in range(0, len(self.reps) - 1, 2):
                self._try_swap(i, i + 1, rng)
            # passe 1-2, 3-4, ...
            for R in self.reps:
                for _ in range(self.sweeps):
                    self._sweep(R)
            for i in range(1, len(self.reps) - 1, 2):
                self._try_swap(i, i + 1, rng)
        return self.best, self.best_E


# ------------------ Visualisation : PNG 3D --------------------


def plot_hp_lattice_3d(
    xyz, hp, E=None, title="Meilleure conformation (3D)", point_size=500,
    line_w=2.0
):
    """
    Affiche une conformation 3D : lignes noires + points colorés H/P.
    """
    xs, ys, zs = zip(*xyz)
    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    # Liaisons
    ax.plot(xs, ys, zs, lw=line_w, color="black", zorder=1)
    # Atomes (résidus)
    cols = [COLORS[t] for t in hp]
    ax.scatter(xs, ys, zs, s=point_size, c=cols, edgecolor="k",
               depthshade=True)

    # Limites + aspect égal
    pad = 0.6
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    zmin, zmax = min(zs) - pad, max(zs) + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1))

    ax.set_title(f"{title} (E={E})" if E is not None else title, pad=10)
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["H"],
            markeredgecolor="black",
            markersize=9,
            label="H",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["P"],
            markeredgecolor="black",
            markersize=9,
            label="P",
        ),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()
    return fig, ax


# ------------------ Trajectoire (frames + énergies) -----------


def run_mc_record(protein, steps=2000, every=1, temperature=1.0,
                  seed=MASTER_SEED):
    """
    Enregistre une trajectoire pour la vidéo.
    À chaque *frame*, on effectue 'every' tentatives Metropolis à T.
    Pour **toutes** les itérations dans la vidéo, mets every=1.
    """
    rng = random.Random(seed)
    T = float(temperature)
    frames = []
    energies = []
    hp = protein.sequence_hp

    E = protein.energy()
    frames.append([tuple(r.coord) for r in protein.residues])
    energies.append(E)

    for _ in range(steps):
        for __ in range(every):
            cand = protein.copy()
            if cand.pull_move_inplace(rng):
                E2 = cand.energy()
                dE = E2 - E
                if dE <= 0 or rng.random() < math.exp(-dE / T):
                    protein, E = cand, E2
        frames.append([tuple(r.coord) for r in protein.residues])
        energies.append(E)

    return frames, energies, hp


# ------------------ Vidéo 3D (MP4/GIF) ------------------------


def save_slider_like_mp4_3d(
    frames,
    energies,
    hp,
    out_path,
    fps=12,
    dpi=120,
    bitrate=2400,
    point_size=30,
    auto_open=True,
):
    """
    Vidéo 2 panneaux :
      - Gauche : conformation 3D (ligne noire + petits points H/P)
      - Droite : courbe d'énergie complète (gris) + point rouge pour la frame 
      courante
    """
    if len(frames) != len(energies):
        raise ValueError(
            f"frames et energies doivent avoir la même longueur "
            f"(frames={len(frames)}, energies={len(energies)})"
        )
    n = len(frames)
    xs_steps = list(range(n))

    fig = plt.figure(figsize=(12.5, 6.2), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
    axL = fig.add_subplot(gs[0, 0], projection="3d")
    axR = fig.add_subplot(gs[0, 1])
    fig.suptitle("Monte Carlo 3D : conformation (gauche) + énergie (droite)")

    # Prépare limites globales pour un cadrage stable
    xs_all = [p[0] for f in frames for p in f]
    ys_all = [p[1] for f in frames for p in f]
    zs_all = [p[2] for f in frames for p in f]
    pad = 0.6
    xmin, xmax = min(xs_all) - pad, max(xs_all) + pad
    ymin, ymax = min(ys_all) - pad, max(ys_all) + pad
    zmin, zmax = min(zs_all) - pad, max(zs_all) + pad
    axL.set_xlim(xmin, xmax)
    axL.set_ylim(ymin, ymax)
    axL.set_zlim(zmin, zmax)
    axL.set_box_aspect((xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1))

    # Panneau gauche : structure 3D
    xs0, ys0, zs0 = zip(*frames[0])
    (lineL,) = axL.plot(xs0, ys0, zs0, lw=2.0, color="black", zorder=1)
    scatL = axL.scatter(
        xs0,
        ys0,
        zs0,
        s=point_size,
        c=[COLORS[t] for t in hp],
        edgecolor="k",
        depthshade=True,
    )
    txtL = axL.text2D(0.02, 0.98, "", transform=axL.transAxes, va="top",
                      ha="left")

    # Panneau droit : énergie
    axR.plot(xs_steps, energies, lw=1.3, color="#9aa0a6", alpha=0.9,
             label="Énergie")
    red_dot = axR.scatter([xs_steps[0]], [energies[0]], s=40, c="red",
                          zorder=3)
    axR.grid(True, alpha=0.3)
    axR.set_xlabel("Itération (frame)")
    axR.set_ylabel("Énergie")
    axR.legend(loc="best")
    emin, emax = min(energies), max(energies)
    if emin == emax:
        emin -= 1
        emax += 1
    m = 0.05 * (emax - emin)
    axR.set_ylim(emin - m, emax + m)
    axR.set_xlim(xs_steps[0], xs_steps[-1])
    txtR = axR.text(0.98, 0.98, "", transform=axR.transAxes, va="top",
                    ha="right")

    def _update(i):
        xs, ys, zs = zip(*frames[i])
        lineL.set_data(xs, ys)
        lineL.set_3d_properties(zs)
        scatL._offsets3d = (xs, ys, zs)
        txtL.set_text(f"t={xs_steps[i]}   E={energies[i]}")
        red_dot.set_offsets([[xs_steps[i], energies[i]]])
        txtR.set_text(f"E={energies[i]}")
        return lineL, scatL, red_dot, txtL, txtR

    ani = animation.FuncAnimation(fig, _update, frames=n, blit=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".mp4" and \
            animation.writers.is_available("ffmpeg"):
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=bitrate,
            extra_args=QT_ARGS,
            metadata={"artist": "HP-folding-3D"},
        )
        ani.save(str(out_path), writer=writer, dpi=dpi)
        saved = out_path
    else:
        gif = out_path.with_suffix(".gif")
        ani.save(str(gif), writer=PillowWriter(fps=fps), dpi=dpi)
        saved = gif

    plt.close(fig)

    if auto_open:
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(saved)], check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", str(saved)], check=False)
            elif sys.platform.startswith("win"):
                subprocess.run(["start", str(saved)], shell=True, check=False)
        except Exception:
            pass

    return saved


# ---------- Saisie (optionnelle) d'une séquence AA ----------


def ask_yes_no(question: str, default: str = "n") -> bool:
    """Boucle jusqu'à 'o'/'n' (ou Entrée -> défaut)."""
    default = "o" if default.lower().startswith("o") else "n"
    prompt = f"{question} (o/n) [{'o' if default=='o' else 'n'}] : "
    while True:
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            print(
                "\n[Non interactif] Réponse par défaut:",
                "oui" if default == "o" else "non",
            )
            return default == "o"
        if ans == "":
            return default == "o"
        if ans in {"o", "oui", "y", "yes"}:
            return True
        if ans in {"n", "non", "no"}:
            return False
        print("Merci de répondre par 'o' (oui) ou 'n' (non).")


def ask_sequence(default_seq="HSQGTFTSDYSKYLDSRRAQDFVQWLMNT"):
    """
    Propose de saisir une séquence AA. Valide les lettres (20 AA de HP_MAP).
    Si refus ou invalide répété -> glucagon par défaut.
    """
    allowed = set(HP_MAP.keys())
    if ask_yes_no(
        "Voulez-vous saisir votre propre séquence protéique en AA ?",
        default="n"
    ):
        while True:
            try:
                s = (
                    input("Entrez une chaîne AA sans espaces (ex: ACDE...): ")
                    .strip()
                    .upper()
                    .replace(" ", "")
                )
            except EOFError:
                print("\n[Non interactif] Séquence par défaut utilisée.")
                s = ""
            if s and all(ch in allowed for ch in s):
                print(f"Séquence fournie : {s} (longueur = {len(s)} AA)")
                return s
            print("Séquence invalide. Lettres autorisées :",
                  "".join(sorted(allowed)))
    print(
        f"Séquence par défaut : glucagon = {default_seq} (longueur = "
        f"{len(default_seq)} AA)"
    )
    return default_seq


# ------------------ Programme principal -----------------------


if __name__ == "__main__":
    # Séquence AA
    aa_seq = ask_sequence()
    seq_hp = conversion_hp(aa_seq)

    out_dir = Path("out3d")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== 1) Monte Carlo simple (vidéo) =====
    P_for_video = Protein(seq_hp)
    frames, energies, hp = run_mc_record(
        P_for_video,
        steps=2000,  # nombre de frames - 1
        every=1,  # 1 = chaque itération devient une frame
        temperature=1.0,
    )
    best_E_mc = min(energies)
    print("[MC 3D] Meilleure énergie :", best_E_mc)
    video_path = save_slider_like_mp4_3d(
        frames,
        energies,
        hp,
        out_path=out_dir / "traj_3d_slider_like.mp4",
        fps=24,
        point_size=30,  # à ajuster selon taille
    )
    print("[MC 3D] Vidéo enregistrée : dossier out3d/")

    # ===== 2) Replica Exchange (sweep = N tentatives) =====
    temps = geometric_temps(t_min=0.5, t_max=6.0, n=12)
    RE = ReplicaExchange(
        Protein(seq_hp),
        temps=temps,
        sweeps=200,  # 200*N tentatives par phase, à ajuster selon taille
        exchanges=60,  # 60 cycles (sweep + échanges alternés), à ajuster
    )
    best3D, Ebest3D = RE.run()
    print("[MC+RE 3D] Meilleure énergie :", Ebest3D)

    # ===== 3) PNG de la meilleure conformation trouvée =====
    xyz = [r.coord for r in best3D.residues]
    fig, ax = plot_hp_lattice_3d(
        xyz,
        best3D.sequence_hp,
        E=Ebest3D,
        title="Meilleure conformation (MC + RE) - 3D",
        point_size=500,
    )  # à ajuster selon sequence
    fig.savefig(out_dir / "best_3d.png", dpi=160, bbox_inches="tight")
    plt.show()
    print("[MC+RE 3D] Image enregistrée : dossier out3d/")
