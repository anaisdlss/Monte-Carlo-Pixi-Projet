"""Anaïs DELASSUS - M2BI"""

import math
import random
import sys
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Circle

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


# ------------------ Outils géométriques (2D) ------------------


def l1_adjacent(p, q):
    """Vrai si p et q voisins sur la grille (distance de Manhattan = 1)."""
    return abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1


def add(p, q):
    """Ajoute des vecteurs."""
    return (p[0] + q[0], p[1] + q[1])


def sub(p, q):
    """Trouve une direction."""
    return (p[0] - q[0], p[1] - q[1])


# ------------------ Modèle HP 2D (pull moves) -----------------


class Residue:
    """Un résidu H/P placé sur la grille 2D."""

    def __init__(self, typ, coord):
        assert typ in ("H", "P")
        self.type = typ
        self.coord = (int(coord[0]), int(coord[1]))


class Protein:
    """
    Chaîne HP sur réseau 2D. Les coordonnées sont des entiers.
    L'état initial est une ligne horizontale.
    """

    def __init__(self, sequence_hp):
        if not sequence_hp or any(c not in ("H", "P") for c in sequence_hp):
            raise ValueError("Séquence HP invalide (uniquement 'H' et 'P').")
        self.sequence_hp = sequence_hp
        self.residues = [Residue(t, (i, 0)) for i, t in enumerate(sequence_hp)]

    # ------ validité + énergie ------
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
        Énergie HP standard : -1 par contact H-H non consécutif (comptés une
        seule fois).
        """
        occ = {r.coord: i for i, r in enumerate(self.residues)}
        E = 0
        for i, r in enumerate(self.residues):
            if r.type != "H":
                continue
            for ax in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                npos = add(r.coord, ax)
                j = occ.get(npos)
                if j is not None and self.residues[j].type == "H" and abs(i - j) > 1:
                    E -= 1
        return E // 2  # chaque contact vu 2 fois

    def copy(self):
        """Copie les infos residues de la protéine."""
        p = Protein(self.sequence_hp)
        p.residues = [Residue(r.type, r.coord) for r in self.residues]
        return p

    # ------ pull move (corner / simple / propagé) ------
    def pull_move_inplace(self, rng):
        """
        Tente un pull move :
          1) Tire un index k au hasard.
          2) Choisit une direction perpendiculaire au segment (k et k±1).
          3) Gère corner, simple et propagé.
        Applique UNE candidate valide tirée au sort. Renvoie True si appliqué.
        """
        n = len(self.residues)
        if n < 3:
            return False  # si chaine trop courte pas de pull move

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
            v = sub(D, A)
            perp = [(0, 1), (0, -1)] if v[0] != 0 else [(1, 0), (-1, 0)]

            for pdir in perp:
                L = add(D, pdir)  # nouvelle case pour A
                C = add(A, pdir)  # coin
                if L in occ:
                    continue

                # --- corner : l’upstream (i-1) occupe déjà le coin ---
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
    """Monte Carlo Metropolis basique sur le modèle HP 2D, avec pull moves."""

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
    Parallel Tempering :
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

    def __init__(self, protein, temps=None, sweeps=200, exchanges=60, seed=MASTER_SEED):
        self.temps = temps or geometric_temps()
        self.reps = [self._Rep(protein, T, seed + i)
                     for i, T in enumerate(self.temps)]
        self.sweeps = int(sweeps)
        self.exchanges = int(exchanges)
        # global best
        self.best = self.reps[0].best.copy()
        self.best_E = self.reps[0].best_E

    def _one_move(self, R):
        """Tente 1 pull move sur la réplique R (Metropolis à T=R.T)."""
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


# ------------------ Visualisation : PNG 2D --------------------


def plot_hp_lattice_2d(
    xy, hp, E=None, title="Meilleure conformation MC + RE", r=0.40, pad=0.20, line_w=2.5
):
    """
    Affiche une conformation 2D : cercles H/P (petits) + liaisons noires.
    r plus petit pour bien voir les liaisons.
    """
    xs, ys = zip(*xy)
    xc = [x + 0.5 for x in xs]
    yc = [y + 0.5 for y in ys]
    xmin, xmax = min(xs) - pad, max(xs) + 1 + pad
    ymin, ymax = min(ys) - pad, max(ys) + 1 + pad

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    ax.plot(xc, yc, lw=line_w, color="black", zorder=1)
    for (cx, cy), t in zip(zip(xc, yc), hp):
        ax.add_patch(
            Circle(
                (cx, cy),
                r,
                facecolor=COLORS[t],
                edgecolor="black",
                linewidth=0.9,
                zorder=2,
            )
        )
    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(math.floor(min(xs)), math.ceil(max(xs)) + 2))
    ax.set_yticks(range(math.floor(min(ys)), math.ceil(max(ys)) + 2))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True, linewidth=0.6, alpha=0.35)
    for s in ax.spines.values():
        s.set_visible(False)
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


def run_mc_record(protein, steps=2000, every=1, temperature=1.0, seed=MASTER_SEED):
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


# ------------------ Vidéo (MP4/GIF) ------------


def save_slider_like_mp4_2d(
    frames,
    energies,
    hp,
    out_path,
    fps=12,
    dpi=120,
    bitrate=2400,
    point_size=40,
    auto_open=True,
):
    """
    Vidéo 2 panneaux :
      - Gauche : conformation (ligne noire + petits points H/P)
      - Droite : courbe d'énergie complète (gris) + **point rouge** pour la
      frame courante
    Plus lent (fps=12) pour bien voir la dynamique.
    """
    if len(frames) != len(energies):
        raise ValueError(
            f"frames et energies doivent avoir la même longueur "
            f"(frames={len(frames)}, energies={len(energies)})"
        )
    n = len(frames)
    xs_steps = list(range(n))

    fig = plt.figure(figsize=(11.5, 5.8), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    fig.suptitle("Monte Carlo : conformation (gauche) + énergie (droite)")

    # panneau gauche
    xs0, ys0 = zip(*frames[0])
    xc0 = [x + 0.5 for x in xs0]
    yc0 = [y + 0.5 for y in ys0]
    (lineL,) = axL.plot(xc0, yc0, lw=2.2, color="black", zorder=1)
    scatL = axL.scatter(
        xc0, yc0, s=point_size, c=[COLORS[t] for t in hp], edgecolors="k", zorder=2
    )
    xs_all = [p[0] for f in frames for p in f]
    ys_all = [p[1] for f in frames for p in f]
    pad = 0.6
    axL.set_xlim(min(xs_all) - pad, max(xs_all) + 1 + pad)
    axL.set_ylim(min(ys_all) - pad, max(ys_all) + 1 + pad)
    axL.set_aspect("equal")
    axL.grid(True, alpha=0.35, linewidth=0.6)
    for sp in axL.spines.values():
        sp.set_visible(False)
    txtL = axL.text(0.02, 0.98, "", transform=axL.transAxes,
                    va="top", ha="left")

    # panneau droit : courbe grise + point rouge courant
    axR.plot(xs_steps, energies, lw=1.3,
             color="#9aa0a6", alpha=0.9, label="Énergie")
    red_dot = axR.scatter([xs_steps[0]], [energies[0]],
                          s=40, c="red", zorder=3)
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
    txtR = axR.text(0.98, 0.98, "", transform=axR.transAxes,
                    va="top", ha="right")

    def _update(i):
        xs, ys = zip(*frames[i])
        xc = [x + 0.5 for x in xs]
        yc = [y + 0.5 for y in ys]
        lineL.set_data(xc, yc)
        scatL.set_offsets(list(zip(xc, yc)))
        txtL.set_text(f"t={xs_steps[i]}   E={energies[i]}")
        red_dot.set_offsets([[xs_steps[i], energies[i]]])
        txtR.set_text(f"E={energies[i]}")
        return lineL, scatL, red_dot, txtL, txtR

    ani = animation.FuncAnimation(fig, _update, frames=n, blit=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".mp4" and animation.writers.is_available("ffmpeg"):
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=bitrate,
            extra_args=QT_ARGS,
            metadata={"artist": "HP-folding"},
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


def ask_sequence(default_seq="HSQGTFTSDYSKYLDSRRAQDFVQWLMNT"):
    """
    Demande à l'utilisateur s'il veut fournir sa propre séquence AA.
    - Si 'o'/'oui' -> lit une séquence et la valide (lettres appartenant à
    HP_MAP).
    - Sinon -> utilise la séquence par défaut (glucagon).
    Affiche toujours la séquence retenue et sa longueur en AA.
    """
    try:
        rep = (
            input(
                "Voulez-vous saisir votre propre séquence protéique en AA ? (o/n) \
                [n] : "
            )
            .strip()
            .lower()
        )
    except EOFError:
        rep = ""  # cas d'exécution non interactive

    if rep in ("o", "oui", "y", "yes"):
        try:
            s = (
                input(
                    "Entrez une chaîne AA (lettres sans espace, ex: ACDE...\
                ):"
                )
                .strip()
                .upper()
                .replace(" ", "")
            )
        except EOFError:
            s = ""

        if s and all(ch in HP_MAP for ch in s):
            print(f"Séquence fournie : {s} (longueur = {len(s)} AA)")
            return s
        else:
            print("Séquence invalide ou vide. Utilisation du glucagon par " "défaut.")

    # défaut : glucagon
    print(
        f"Séquence par défaut: glucagon={default_seq}(longueur={len(
            default_seq)} AA)"
    )
    return default_seq


# ------------------ Programme principal -----------------------


if __name__ == "__main__":
    # Séquence AA : glucagon (chaîne unique, courte)
    aa_seq = ask_sequence()
    seq_hp = conversion_hp(aa_seq)

    out_dir = Path("out2d")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== 1) Monte Carlo simple (vidéo) =====
    P_for_video = Protein(seq_hp)
    frames, energies, hp = run_mc_record(
        P_for_video,
        steps=2000,  # nombre de frames - 1
        # 1 = chaque itération devient une frame (demande de “tout voir”)
        every=1,
        temperature=1.0,
    )
    best_E_mc = min(energies)
    print("[MC 2D] Meilleure énergie :", best_E_mc)
    video_path = save_slider_like_mp4_2d(
        frames,
        energies,
        hp,
        out_path=out_dir / "traj_2d_slider_like.mp4",
        fps=24,
        point_size=40,
    )
    print("[MC 2D] Vidéo enregistrée : dossier out2d/")

    # ===== 2) Replica Exchange (sweep = N tentatives) =====
    # ladder géométrique “classique”
    temps = geometric_temps(t_min=0.5, t_max=6.0, n=10)
    RE = ReplicaExchange(
        Protein(seq_hp),
        temps=temps,
        sweeps=200,  # ici 200*N tentatives par phase
        exchanges=60,  # 60 cycles (sweep + échanges alternés)
    )
    best2D, Ebest2D = RE.run()
    print("[MC+RE 2D] Meilleure énergie :", Ebest2D)

    # ===== 3) PNG de la meilleure conformation trouvée =====
    xy = [r.coord for r in best2D.residues]
    fig, ax = plot_hp_lattice_2d(
        xy, best2D.sequence_hp, E=Ebest2D, title="Meilleure conformation \
            (MC + RE)"
    )
    fig.savefig(out_dir / "best_2d.png", dpi=160, bbox_inches="tight")
    plt.show()
    print("[MC+RE 2D] Image enregistrée : dossier out2d/")
