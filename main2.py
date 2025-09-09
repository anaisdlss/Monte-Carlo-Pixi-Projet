"""
HP folding sur réseau 2D ou 3D (pull moves only) + Metropolis MC + Replica-Exchange.
Visualisations :
  - Meilleure conformation 2D/3D (PNG)
  - Animation MP4 2D/3D enregistrée SANS affichage
  - Slider interactif pour explorer la trajectoire (par défaut en 2D)

Dépendances Python : matplotlib
Pour MP4 : nécessite FFmpeg installé sur le système (ex: 'pixi add ffmpeg -c conda-forge')
"""

from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Circle
from matplotlib import animation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (nécessaire pour 3D)
import random
import math
import matplotlib.pyplot as plt
import os

# ----------------------
# Conversion AA -> HP
# ----------------------
HP_MAP = {
    "A": "H", "V": "H", "L": "H", "I": "H", "M": "H", "F": "H", "W": "H", "Y": "H",
    "R": "P", "N": "P", "D": "P", "C": "P", "Q": "P", "E": "P", "G": "P", "H": "P",
    "K": "P", "P": "P", "S": "P", "T": "P"
}


def conversion_hp(seq_aa):
    return "".join(HP_MAP[aa] for aa in seq_aa)

# ----------------------
# Outils géométriques
# ----------------------


def l1_adjacent(p, q):
    return sum(abs(a - b) for a, b in zip(p, q)) == 1


def unit_axes(dim):
    axes = []
    for k in range(dim):
        v = [0]*dim
        v[k] = 1
        axes.append(tuple(v))
        v[k] = -1
        axes.append(tuple(v))
    return axes


def add(p, q): return tuple(a + b for a, b in zip(p, q))
def sub(p, q): return tuple(a - b for a, b in zip(p, q))
def dot(p, q): return sum(a*b for a, b in zip(p, q))

# ----------------------
# Données
# ----------------------


class Residue:
    def __init__(self, typ, coord):
        if typ not in ("H", "P"):
            raise ValueError("Residue.type doit être 'H' ou 'P'.")
        self.type = typ
        self.coord = tuple(coord)


class Protein:
    """Chaîne HP sur réseau 2D (carré) ou 3D (cubique), liaisons Manhattan."""

    def __init__(self, sequence_hp, dim=2):
        if dim not in (2, 3):
            raise ValueError("dim doit valoir 2 ou 3.")
        if not sequence_hp or any(c not in ("H", "P") for c in sequence_hp):
            raise ValueError(
                "La séquence HP doit contenir uniquement 'H'/'P'.")
        self.sequence_hp = sequence_hp
        self.dim = dim
        self.residues = []
        for i, t in enumerate(sequence_hp):
            self.residues.append(Residue(t, (i, 0) if dim == 2 else (i, 0, 0)))

    # validité
    def _valid_chain(self, pos):
        if len(pos) != len(set(pos)):
            return False
        for i in range(len(pos)-1):
            if not l1_adjacent(pos[i], pos[i+1]):
                return False
        return True

    # énergie HP
    def energy(self):
        occ = {r.coord: i for i, r in enumerate(self.residues)}
        E = 0
        for i, r in enumerate(self.residues):
            if r.type != "H":
                continue
            for axis in unit_axes(self.dim):
                npos = add(r.coord, axis)
                j = occ.get(npos)
                if j is not None and self.residues[j].type == "H" and abs(i - j) > 1:
                    E -= 1
        return E // 2

    # copie
    def copy(self):
        p = Protein(self.sequence_hp, dim=self.dim)
        p.residues = [Residue(r.type, r.coord) for r in self.residues]
        return p

    # pull move 2D/3D
    def pull_move_inplace(self):
        n = len(self.residues)
        if n < 3:
            return False

        old = [r.coord for r in self.residues]
        occ = {pos: i for i, pos in enumerate(old)}
        k = random.randrange(n)
        candidates = []

        for s in (-1, +1):
            iA = k
            downstream_idx = k - s
            upstream_idx = k + s
            if not (0 <= downstream_idx < n):
                continue

            A = old[iA]
            D = old[downstream_idx]
            if not l1_adjacent(A, D):
                continue
            v = sub(D, A)  # ±axe unitaire

            # vecteurs unitaires perpendiculaires à v
            perp_dirs = [u for u in unit_axes(self.dim) if dot(u, v) == 0]

            for pdir in perp_dirs:
                L = add(D, pdir)   # nouvelle case pour A
                C = add(A, pdir)   # coin

                if L in occ:
                    continue

                # Corner : C occupé par upstream
                if 0 <= upstream_idx < n and occ.get(C) == upstream_idx:
                    new = old[:]
                    new[iA] = L
                    if self._valid_chain(new):
                        candidates.append(new)
                    continue

                # bloqué ou pas d'upstream
                if C in occ or not (0 <= upstream_idx < n):
                    continue

                # Pull simple / propagé
                new = old[:]
                new[iA] = L
                new[upstream_idx] = C

                def check_local(pos, j):
                    return l1_adjacent(pos[j], pos[j - s])

                j = upstream_idx + s
                while 0 <= j < n and not check_local(new, j):
                    src = j - 2*s
                    if not (0 <= src < n):
                        break
                    cand_pos = old[src]
                    # collision
                    if cand_pos in new and new.index(cand_pos) != j:
                        break
                    new[j] = cand_pos
                    j += s

                if self._valid_chain(new):
                    candidates.append(new)

        if not candidates:
            return False

        chosen = random.choice(candidates)
        for coord, r in zip(chosen, self.residues):
            r.coord = coord
        return True

# ----------------------
# Monte-Carlo (Métropolis)
# ----------------------


class MonteCarlo:
    def __init__(self, protein, temperature=1.0, steps=2000, seed=None):
        self.protein = protein
        self.T = float(temperature)
        self.steps = int(steps)
        if seed is not None:
            random.seed(seed)
        self.curr_E = protein.energy()
        self.best = protein.copy()
        self.best_E = self.curr_E

    @staticmethod
    def _metropolis(dE, T):
        return dE <= 0 or random.random() < math.exp(-dE / T)

    def run(self):
        for _ in range(self.steps):
            cand = self.protein.copy()
            if not cand.pull_move_inplace():
                continue
            E_cand = cand.energy()
            if self._metropolis(E_cand - self.curr_E, self.T):
                self.protein = cand
                self.curr_E = E_cand
                if self.curr_E < self.best_E:
                    self.best_E = self.curr_E
                    self.best = cand.copy()
        return self.protein, self.best, self.best_E

# ----------------------
# Replica-Exchange (Parallel Tempering)
# ----------------------


def geometric_temps(t_min=0.6, t_max=5.0, n=8):
    if n < 2:
        return [float(t_min)]
    ratio = (t_max / t_min) ** (1.0 / (n - 1))
    return [t_min * (ratio ** k) for k in range(n)]


class ReplicaExchange:
    class _Replica:
        def __init__(self, protein, T):
            self.protein = protein
            self.T = float(T)
            self.E = protein.energy()
            self.best = protein.copy()
            self.best_E = self.E

    def __init__(self, protein, temps=None, sweeps=200, exchanges=200, seed=None):
        if seed is not None:
            random.seed(seed)
        self.temps = temps or geometric_temps()
        self.replicas = [self._Replica(protein.copy(), T) for T in self.temps]
        self.sweeps = int(sweeps)
        self.exchanges = int(exchanges)
        self.best = self.replicas[0].best.copy()
        self.best_E = self.replicas[0].best_E
        self._refresh_global_best()

    @staticmethod
    def _metropolis(dE, T):
        return dE <= 0 or random.random() < math.exp(-dE / T)

    def _mc_sweep(self, R):
        for _ in range(self.sweeps):
            cand = R.protein.copy()
            if not cand.pull_move_inplace():
                continue
            E_new = cand.energy()
            if self._metropolis(E_new - R.E, R.T):
                R.protein = cand
                R.E = E_new
                if R.E < R.best_E:
                    R.best_E = R.E
                    R.best = R.protein.copy()
                if R.E < self.best_E:
                    self.best_E = R.E
                    self.best = R.protein.copy()

    def _attempt_swap(self, i, j):
        Ri, Rj = self.replicas[i], self.replicas[j]
        beta_i, beta_j = 1.0 / Ri.T, 1.0 / Rj.T
        delta = (beta_i - beta_j) * (Rj.E - Ri.E)
        if delta >= 0 or random.random() < math.exp(delta):
            Ri.protein, Rj.protein = Rj.protein, Ri.protein
            Ri.E, Rj.E = Rj.E, Ri.E
            if Ri.E < self.best_E:
                self.best_E = Ri.E
                self.best = Ri.protein.copy()
            if Rj.E < self.best_E:
                self.best_E = Rj.E
                self.best = Rj.protein.copy()

    def _swap_pass(self, offset):
        for i in range(offset, len(self.replicas) - 1, 2):
            self._attempt_swap(i, i + 1)

    def _refresh_global_best(self):
        for R in self.replicas:
            if R.E < self.best_E:
                self.best_E = R.E
                self.best = R.protein.copy()

    def run(self):
        for _ in range(self.exchanges):
            for R in self.replicas:
                self._mc_sweep(R)
            self._swap_pass(offset=0)
            for R in self.replicas:
                self._mc_sweep(R)
            self._swap_pass(offset=1)
        return self.best, self.best_E

# ----------------------
# Visualisation 2D
# ----------------------


def plot_hp_lattice_2d(xy, hp, E=None, title="Meilleure conformation (2D)",
                       r=0.40, pad=0.20, line_w=2.5):
    xs, ys = zip(*xy)
    xc = [x + 0.5 for x in xs]
    yc = [y + 0.5 for y in ys]
    xmin, xmax = min(xs) - pad, max(xs) + 1 + pad
    ymin, ymax = min(ys) - pad, max(ys) + 1 + pad

    colors = {'H': '#00cfe6', 'P': '#8c5a2b'}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    ax.plot(xc, yc, lw=line_w, color='black', zorder=1)
    for (cx, cy), t in zip(zip(xc, yc), hp):
        ax.add_patch(Circle((cx, cy), r, facecolor=colors[t],
                            edgecolor='black', linewidth=0.9, zorder=2))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(math.floor(min(xs)), math.ceil(max(xs)) + 2))
    ax.set_yticks(range(math.floor(min(ys)), math.ceil(max(ys)) + 2))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True, which='both', linewidth=0.6, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{title} (E = {E})" if E is not None else title, pad=10)
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
# Visualisation 3D
# ----------------------


def plot_hp_lattice_3d(xyz, hp, E=None, title="Meilleure conformation (3D)",
                       s=120, line_w=2.0, pad=0.8):
    xs, ys, zs = zip(*xyz)
    xc = [x + 0.5 for x in xs]
    yc = [y + 0.5 for y in ys]
    zc = [z + 0.5 for z in zs]

    colors = {'H': '#00cfe6', 'P': '#8c5a2b'}
    cols = [colors[t] for t in hp]

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    # liaisons
    for i in range(len(xc) - 1):
        ax.plot([xc[i], xc[i+1]], [yc[i], yc[i+1]], [zc[i], zc[i+1]],
                lw=line_w, color='black', alpha=0.9, zorder=1)

    # résidus
    ax.scatter(xc, yc, zc, s=s, c=cols, edgecolors='k',
               depthshade=True, zorder=2)

    xmin, xmax = min(xs) - pad, max(xs) + 1 + pad
    ymin, ymax = min(ys) - pad, max(ys) + 1 + pad
    zmin, zmax = min(zs) - pad, max(zs) + 1 + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))
    ax.set_title(f"{title} (E = {E})" if E is not None else title, pad=10)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    h_proxy = Line2D([0], [0], marker='o', color='none', markerfacecolor=colors['H'],
                     markeredgecolor='black', markersize=8, label='H')
    p_proxy = Line2D([0], [0], marker='o', color='none', markerfacecolor=colors['P'],
                     markeredgecolor='black', markersize=8, label='P')
    ax.legend(handles=[h_proxy, p_proxy], loc='upper left')

    plt.tight_layout()
    return fig, ax

# ----------------------
# Trajectoires + Animations + Sliders
# ----------------------


def run_mc_record(protein, steps=3000, every=10, temperature=1.0, seed=None):
    """
    Lance un MC Metropolis et enregistre la conformation toutes les 'every' itérations.
    Retourne: frames (liste de listes de coords), energies (liste d'int), hp (str)
    """
    if seed is not None:
        random.seed(seed)
    T = float(temperature)
    frames = []
    energies = []
    hp = protein.sequence_hp

    # état initial
    frames.append([tuple(r.coord) for r in protein.residues])
    energies.append(protein.energy())

    for t in range(steps):
        cand = protein.copy()
        if cand.pull_move_inplace():
            E_new = cand.energy()
            dE = E_new - protein.energy()
            if dE <= 0 or random.random() < math.exp(-dE / T):
                protein = cand
        if (t + 1) % every == 0:
            frames.append([tuple(r.coord) for r in protein.residues])
            energies.append(protein.energy())
    return frames, energies, hp


def _setup_ax_2d(frames, hp, title="Animation 2D"):
    xy0 = frames[0]
    xs, ys = zip(*xy0)
    xc = [x + 0.5 for x in xs]
    yc = [y + 0.5 for y in ys]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    line, = ax.plot(xc, yc, lw=2.0, color="black", zorder=1)
    scat = ax.scatter(xc, yc, s=220, edgecolors="k", zorder=2)

    cols = {"H": "#00cfe6", "P": "#8c5a2b"}
    scat.set_facecolors([cols[t] for t in hp])

    xs_all = [p[0] for f in frames for p in f]
    ys_all = [p[1] for f in frames for p in f]
    pad = 0.6
    ax.set_xlim(min(xs_all) - pad, max(xs_all) + 1 + pad)
    ax.set_ylim(min(ys_all) - pad, max(ys_all) + 1 + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35, linewidth=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)
    return fig, ax, line, scat


def _update_2d(i, frames, line, scat):
    xy = frames[i]
    xs, ys = zip(*xy)
    xc = [x + 0.5 for x in xs]
    yc = [y + 0.5 for y in ys]
    line.set_data(xc, yc)
    scat.set_offsets(list(zip(xc, yc)))
    return line, scat


def save_animation_mp4_2d(frames, hp, out_path="traj_2d.mp4", fps=25):
    """Enregistre une animation 2D en MP4 sans afficher la figure."""
    fig, ax, line, scat = _setup_ax_2d(frames, hp, title="HP folding 2D")
    ani = animation.FuncAnimation(fig, _update_2d, frames=len(frames),
                                  fargs=(frames, line, scat), blit=False)
    Writer = animation.writers.get("ffmpeg", None)
    if Writer is None:
        raise RuntimeError(
            "FFmpeg introuvable. Installe-le (conda/pixi/brew/apt).")
    writer = Writer(fps=fps, metadata={"artist": "HP-folding"})
    ani.save(out_path, writer=writer, dpi=120)
    plt.close(fig)


def save_animation_mp4_3d(frames, hp, out_path="traj_3d.mp4", fps=20):
    """Animation 3D MP4 (plus lourde), sans fenêtre."""
    xyz0 = frames[0]
    xs, ys, zs = zip(*xyz0)
    xc = [x + 0.5 for x in xs]
    yc = [y + 0.5 for y in ys]
    zc = [z + 0.5 for z in zs]

    fig = plt.figure(figsize=(7, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    line, = ax.plot(xc, yc, zc, lw=2.0, color="black", zorder=1)
    cols = {"H": "#00cfe6", "P": "#8c5a2b"}
    scat = ax.scatter(xc, yc, zc, s=120, c=[
                      cols[t] for t in hp], edgecolors="k", depthshade=True, zorder=2)

    xs_all = [p[0] for f in frames for p in f]
    ys_all = [p[1] for f in frames for p in f]
    zs_all = [p[2] for f in frames for p in f]
    pad = 0.8
    ax.set_xlim(min(xs_all) - pad, max(xs_all) + 1 + pad)
    ax.set_ylim(min(ys_all) - pad, max(ys_all) + 1 + pad)
    ax.set_zlim(min(zs_all) - pad, max(zs_all) + 1 + pad)
    ax.set_box_aspect((ax.get_xlim()[1]-ax.get_xlim()[0],
                       ax.get_ylim()[1]-ax.get_ylim()[0],
                       ax.get_zlim()[1]-ax.get_zlim()[0]))
    ax.set_title("HP folding 3D")

    def _update_3d(i):
        xyz = frames[i]
        xs, ys, zs = zip(*xyz)
        xc = [x + 0.5 for x in xs]
        yc = [y + 0.5 for y in ys]
        zc = [z + 0.5 for z in zs]
        line.set_data(xc, yc)
        line.set_3d_properties(zc)
        # recrée le scatter (limitation 3D)
        ax.collections.remove(scat)
        new_scat = ax.scatter(xc, yc, zc, s=120, c=[
                              cols[t] for t in hp], edgecolors="k", depthshade=True, zorder=2)
        return line, new_scat

    ani = animation.FuncAnimation(
        fig, _update_3d, frames=len(frames), blit=False)
    Writer = animation.writers.get("ffmpeg", None)
    if Writer is None:
        raise RuntimeError(
            "FFmpeg introuvable. Installe-le (conda/pixi/brew/apt).")
    writer = Writer(fps=fps, metadata={"artist": "HP-folding"})
    ani.save(out_path, writer=writer, dpi=120)
    plt.close(fig)


def show_slider_2d(frames, hp, energies=None, title="Slider 2D (HP folding)"):
    fig, ax, line, scat = _setup_ax_2d(frames, hp, title=title)
    ax_slider = plt.axes([0.15, 0.02, 0.7, 0.03])
    sidx = Slider(ax_slider, "frame", 0, len(frames)-1, valinit=0, valstep=1)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    def _update(i):
        _update_2d(i, frames, line, scat)
        if energies is not None:
            txt.set_text(f"t={i}  E={energies[i]}")
        else:
            txt.set_text(f"t={i}")
        fig.canvas.draw_idle()

    sidx.on_changed(lambda val: _update(int(val)))
    _update(0)
    plt.show()


def show_slider_3d(frames, hp, energies=None, title="Slider 3D (HP folding)"):
    # Setup 3D proche de save_animation_mp4_3d
    xyz0 = frames[0]
    xs, ys, zs = zip(*xyz0)
    xc = [x + 0.5 for x in xs]
    yc = [y + 0.5 for y in ys]
    zc = [z + 0.5 for z in zs]
    fig = plt.figure(figsize=(7, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    line, = ax.plot(xc, yc, zc, lw=2.0, color="black", zorder=1)
    cols = {"H": "#00cfe6", "P": "#8c5a2b"}
    scat = ax.scatter(xc, yc, zc, s=120, c=[
                      cols[t] for t in hp], edgecolors="k", depthshade=True, zorder=2)

    xs_all = [p[0] for f in frames for p in f]
    ys_all = [p[1] for f in frames for p in f]
    zs_all = [p[2] for f in frames for p in f]
    pad = 0.8
    ax.set_xlim(min(xs_all) - pad, max(xs_all) + 1 + pad)
    ax.set_ylim(min(ys_all) - pad, max(ys_all) + 1 + pad)
    ax.set_zlim(min(zs_all) - pad, max(zs_all) + 1 + pad)
    ax.set_box_aspect((ax.get_xlim()[1]-ax.get_xlim()[0],
                       ax.get_ylim()[1]-ax.get_ylim()[0],
                       ax.get_zlim()[1]-ax.get_zlim()[0]))
    ax.set_title(title)
    txt = ax.text2D(0.02, 0.98, "", transform=ax.transAxes,
                    va="top", ha="left")

    ax_slider = plt.axes([0.15, 0.02, 0.7, 0.03])
    sidx = Slider(ax_slider, "frame", 0, len(frames)-1, valinit=0, valstep=1)

    def _draw(i):
        xyz = frames[i]
        xs, ys, zs = zip(*xyz)
        xc = [x + 0.5 for x in xs]
        yc = [y + 0.5 for y in ys]
        zc = [z + 0.5 for z in zs]
        line.set_data(xc, yc)
        line.set_3d_properties(zc)
        ax.collections.remove(scat)
        new_scat = ax.scatter(xc, yc, zc, s=120, c=[
                              cols[t] for t in hp], edgecolors="k", depthshade=True, zorder=2)
        if energies is not None:
            txt.set_text(f"t={i}  E={energies[i]}")
        else:
            txt.set_text(f"t={i}")
        fig.canvas.draw_idle()
        return new_scat

    sidx.on_changed(lambda val: _draw(int(val)))
    _draw(0)
    plt.show()

# ----------------------
# Utilitaires de sauvegarde
# ----------------------


def save_best_png_2d(best_prot, out_path="best_2d.png"):
    xy = [r.coord for r in best_prot.residues]
    fig, ax = plot_hp_lattice_2d(xy, best_prot.sequence_hp, E=best_prot.energy(),
                                 title="2D - Meilleure conformation")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_best_png_3d(best_prot, out_path="best_3d.png"):
    xyz = [r.coord for r in best_prot.residues]
    fig, ax = plot_hp_lattice_3d(xyz, best_prot.sequence_hp, E=best_prot.energy(),
                                 title="3D - Meilleure conformation")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------
# Exemple d'utilisation
# ----------------------
if __name__ == "__main__":
    # === Paramètres rapides (modifie-les si besoin) ===
    aa_seq = "ACDEFGHIKLMNPQRSTVWY"   # 20 AA pour tourner vite
    seq_hp = conversion_hp(aa_seq)

    # Répertoires/fichiers de sortie
    os.makedirs("out", exist_ok=True)
    PNG_2D = "out/best_2d.png"
    PNG_3D = "out/best_3d.png"
    MP4_2D = "out/traj_2d.mp4"
    MP4_3D = "out/traj_3d.mp4"

    # ===== 2D : meilleur état via Replica-Exchange, PNG, trajectoire + MP4, slider =====
    temps2 = geometric_temps(t_min=0.5, t_max=6.0, n=10)
    re2 = ReplicaExchange(Protein(seq_hp, dim=2),
                          temps=temps2, sweeps=200, exchanges=60, seed=42)
    best2D, Ebest2D = re2.run()
    print("[2D-RE] Meilleure énergie :", Ebest2D)
    save_best_png_2d(best2D, out_path=PNG_2D)

    # Trajectoire MC (depuis la chaîne droite pour voir l'évolution)
    P2 = Protein(seq_hp, dim=2)
    frames2, energies2, hp2 = run_mc_record(
        P2, steps=3000, every=10, temperature=1.0, seed=1)
    save_animation_mp4_2d(frames2, hp2, out_path=MP4_2D, fps=25)

    # Affiche slider 2D (interactif)
    show_slider_2d(frames2, hp2, energies=energies2,
                   title="Slider 2D - HP folding")

    # ===== 3D : meilleur état via Replica-Exchange, PNG, trajectoire + MP4 (slider optionnel) =====
    temps3 = geometric_temps(t_min=0.6, t_max=7.0, n=12)
    re3 = ReplicaExchange(Protein(seq_hp, dim=3),
                          temps=temps3, sweeps=200, exchanges=60, seed=123)
    best3D, Ebest3D = re3.run()
    print("[3D-RE] Meilleure énergie :", Ebest3D)
    save_best_png_3d(best3D, out_path=PNG_3D)

    # Trajectoire MC 3D + MP4 (plus lourd)
    P3 = Protein(seq_hp, dim=3)
    frames3, energies3, hp3 = run_mc_record(
        P3, steps=3000, every=10, temperature=1.2, seed=2)
    save_animation_mp4_3d(frames3, hp3, out_path=MP4_3D, fps=20)

    # Slider 3D (optionnel ; commenter si trop lent)
    # show_slider_3d(frames3, hp3, energies=energies3, title="Slider 3D - HP folding")
