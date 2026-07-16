"""Motion discriminators + one-class motion priors (stages 15-17).

  IMM  -- interacting-multiple-model filter (constant-velocity + coordinated-turn
          bank) run over each candidate track's position estimates. Score = the
          fraction of scans spent in the constant-velocity mode (cv_frac): a real
          flight cruises coherently, clutter/noise thrash between modes.

  VAE  -- a one-class sequence VAE trained ONLY on WHACK01 real GA motion
          (per-step speed + turn-rate). Score = reconstruction error: low = the
          motion lies on the manifold of real flights. Never sees track labels.

  SDE  -- a latent neural SDE: the continuous-time, gap-aware version of the VAE.

Paths resolve through utils.io; nothing here reads truth beyond the training
motion (which is real-flight kinematics, not the tracker's labels).
"""
import numpy as np
import pandas as pd

from utils.io import (TRAIN_DATES, TEST_DATE, get_track_points_path,
                      get_tracks_path, get_real_trajectories_path)

DT = 10.0


# ----------------------------------------------------------------- loaders

def load_track_points(date, ids=None):
    """date -> {track_id: dict of per-DETECTED-scan arrays (scan,e,n,range,snr)}.
    Memory-safe chunked read; keeps only detected scans (miss==0)."""
    path = get_track_points_path(date)
    cols = ["track_id", "scan_idx", "est_e", "est_n", "est_range_m", "snr_db", "miss"]
    keep = []
    idset = set(ids) if ids is not None else None
    for ch in pd.read_csv(path, usecols=cols, chunksize=1_000_000):
        ch = ch[ch.miss == 0]
        if idset is not None:
            ch = ch[ch.track_id.isin(idset)]
        keep.append(ch)
    df = pd.concat(keep).sort_values(["track_id", "scan_idx"])
    out = {}
    for tid, g in df.groupby("track_id", sort=False):
        out[int(tid)] = dict(
            scan=g.scan_idx.to_numpy(),
            e=g.est_e.to_numpy(np.float64),
            n=g.est_n.to_numpy(np.float64),
            rng=g.est_range_m.to_numpy(np.float64),
            snr=g.snr_db.to_numpy(np.float64),
        )
    return out


def load_labels(date):
    """track_id -> (label, track_source, n_det, range_median_m) from the summary."""
    df = pd.read_csv(get_tracks_path(date),
                     usecols=["track_id", "label", "track_source", "n_det", "range_median_m"])
    return df.set_index("track_id")


def load_real_motion(dates, max_traj=None):
    """List of (speed_mps, turn_rate_deg_s) per real GA trajectory from WHACK01."""
    seqs = []
    for d in dates:
        f = get_real_trajectories_path(d)
        df = pd.read_csv(f, usecols=["trajectory_id", "sample_idx", "speed_mps", "turn_rate_deg_s"])
        df = df.sort_values(["trajectory_id", "sample_idx"])
        for _, g in df.groupby("trajectory_id", sort=False):
            sp = g.speed_mps.to_numpy(np.float64)
            tr = np.nan_to_num(g.turn_rate_deg_s.to_numpy(np.float64))
            if len(sp) >= 4:
                seqs.append((sp, tr))
            if max_traj and len(seqs) >= max_traj:
                return seqs
    return seqs


def track_speed_turn(e, n, scan):
    """Per-step speed (m/s) and turn rate (deg/s) from a position sequence,
    dt scaled by scan gaps -- same convention as WHACK03 features.py."""
    if len(e) < 3:
        sp = np.array([np.hypot(np.diff(e), np.diff(n)).sum() / DT]) if len(e) == 2 else np.array([0.0])
        return sp, np.array([0.0])
    dt = np.maximum(np.diff(scan), 1) * DT
    seg = np.hypot(np.diff(e), np.diff(n))
    speed = seg / dt
    heading = np.arctan2(np.diff(e), np.diff(n))
    dh = (np.diff(heading) + np.pi) % (2 * np.pi) - np.pi
    turn = np.degrees(dh) / dt[:-1]
    # align: speed has T-1 entries, turn has T-2; pad turn to match speed length
    turn = np.concatenate([turn[:1], turn]) if len(turn) else np.zeros_like(speed)
    return speed, turn


# ----------------------------------------------------------------- IMM

def _ct_F(omega, dt):
    """Coordinated-turn transition for state [E, vE, N, vN] at turn rate omega."""
    if abs(omega) < 1e-6:
        return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], float)
    w = omega
    s, c = np.sin(w * dt), np.cos(w * dt)
    return np.array([
        [1, s / w,        0, -(1 - c) / w],
        [0, c,            0, -s],
        [0, (1 - c) / w,  1, s / w],
        [0, s,            0, c],
    ], float)


class IMM:
    """Fixed-rate IMM-CT bank over a 4-state CV coordinate frame.

    Models: constant-velocity (omega=0) + coordinated turn at +/- omega0.
    Shared 4-state, so mode mixing needs no augmentation.
    """
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], float)

    def __init__(self, omega0_deg_s=3.0, sigma_a=2.0, p_stay=0.90):
        self.omegas = [0.0, np.radians(omega0_deg_s), -np.radians(omega0_deg_s)]
        self.sigma_a = sigma_a
        M = len(self.omegas)
        # sticky Markov transition matrix
        off = (1.0 - p_stay) / (M - 1)
        self.Pi = np.full((M, M), off) + np.eye(M) * (p_stay - off)
        self.M = M

    def _Q(self, dt):
        G = np.array([[dt ** 2 / 2, 0], [dt, 0], [0, dt ** 2 / 2], [0, dt]])
        return G @ (self.sigma_a ** 2 * np.eye(2)) @ G.T

    def _R(self, rng):
        """Converted-measurement noise from range: 50 m down-range, range*0.2 deg cross."""
        cross = max(rng, 1.0) * np.radians(0.2)
        s2 = 0.5 * (50.0 ** 2 + cross ** 2)   # isotropic average of the two
        return np.eye(2) * s2

    def score(self, e, n, rng, scan):
        """Return dict(score, cv_frac, n_switch, entropy). score = mean per-scan
        measurement log-likelihood under the IMM (higher => more target-like)."""
        T = len(e)
        if T < 4:
            return dict(score=-50.0, cv_frac=0.0, n_switch=0, entropy=0.0)
        z = np.column_stack([e, n])
        dts = np.maximum(np.diff(scan), 1) * DT

        # init from first two points
        v0 = (z[1] - z[0]) / dts[0]
        x = np.array([z[0, 0], v0[0], z[0, 1], v0[1]], float)
        P = np.diag([100.0 ** 2, 50.0 ** 2, 100.0 ** 2, 50.0 ** 2])
        xs = [x.copy() for _ in range(self.M)]
        Ps = [P.copy() for _ in range(self.M)]
        mu = np.full(self.M, 1.0 / self.M)

        logLs, modes = [], []
        for k in range(1, T):
            dt = dts[k - 1]
            R = self._R(rng[k]); Q = self._Q(dt)
            # --- interaction / mixing ---
            cbar = self.Pi.T @ mu                        # predicted mode prob
            Wmix = (self.Pi * mu[:, None]) / np.maximum(cbar[None, :], 1e-12)  # w[i,j]
            x0, P0 = [], []
            for j in range(self.M):
                xm = sum(Wmix[i, j] * xs[i] for i in range(self.M))
                Pm = np.zeros((4, 4))
                for i in range(self.M):
                    d = xs[i] - xm
                    Pm += Wmix[i, j] * (Ps[i] + np.outer(d, d))
                x0.append(xm); P0.append(Pm)
            # --- model-matched predict + update ---
            Lam = np.zeros(self.M)
            for j in range(self.M):
                F = _ct_F(self.omegas[j], dt)
                xp = F @ x0[j]
                Pp = F @ P0[j] @ F.T + Q
                S = self.H @ Pp @ self.H.T + R
                innov = z[k] - self.H @ xp
                Sinv = np.linalg.inv(S)
                K = Pp @ self.H.T @ Sinv
                xs[j] = xp + K @ innov
                Ps[j] = Pp - K @ self.H @ Pp
                d2 = float(innov @ Sinv @ innov)
                Lam[j] = np.exp(-0.5 * d2) / (2 * np.pi * np.sqrt(np.linalg.det(S)))
            # --- mode update + combined likelihood ---
            post = Lam * cbar
            L = post.sum()
            logLs.append(np.log(max(L, 1e-300)))
            mu = post / max(L, 1e-300)
            modes.append(int(np.argmax(mu)))

        logLs = np.array(logLs)
        modes = np.array(modes)
        # warm-up: drop the first 2 steps
        core = logLs[2:] if len(logLs) > 2 else logLs
        p = np.bincount(modes, minlength=self.M) / len(modes)
        ent = float(-(p * np.log(p + 1e-12)).sum())
        return dict(
            score=float(core.mean()),
            cv_frac=float((modes == 0).mean()),
            n_switch=int((np.diff(modes) != 0).sum()),
            entropy=ent,
        )


# ----------------------------------------------------------------- VAE

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIN = 10                      # window length (steps of 10 s = 100 s)
SPEED_SCALE = 100.0          # m/s -> O(1)
TURN_SCALE = 10.0           # deg/s -> O(1)


def windows_from_seq(speed, turn, L=WIN, stride=1):
    """Sliding [speed_norm, turn_norm] windows; pads short sequences by edge-repeat."""
    s = np.asarray(speed, float) / SPEED_SCALE
    t = np.clip(np.asarray(turn, float) / TURN_SCALE, -3, 3)
    x = np.column_stack([s, t])
    if len(x) < L:
        x = np.vstack([x, np.repeat(x[-1:], L - len(x), axis=0)])
    return np.stack([x[i:i + L] for i in range(0, len(x) - L + 1, stride)]).astype(np.float32)


class TrajVAE(nn.Module):
    def __init__(self, n_ch=2, hidden=32, latent=8):
        super().__init__()
        self.enc = nn.GRU(n_ch, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, latent)
        self.lv = nn.Linear(hidden, latent)
        self.dec_in = nn.Linear(latent, hidden)
        self.dec = nn.GRU(n_ch, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_ch)
        self.n_ch = n_ch

    def forward(self, x):
        _, h = self.enc(x)
        h = h[-1]
        mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        h0 = torch.tanh(self.dec_in(z)).unsqueeze(0)
        # teacher-force with a zero-shifted input (autoregressive-lite)
        dec_in = torch.zeros_like(x)
        dec_in[:, 1:] = x[:, :-1]
        y, _ = self.dec(dec_in, h0)
        return self.out(y), mu, lv

    def recon_error(self, x):
        y, mu, lv = self.forward(x)
        return ((y - x) ** 2).mean(dim=(1, 2))


def train_vae(windows, epochs=12, batch=1024, seed=0, beta=0.5):
    torch.manual_seed(seed)
    X = torch.from_numpy(windows).to(DEVICE)
    model = TrajVAE(n_ch=windows.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = len(X); idx = np.arange(n)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        rng.shuffle(idx)
        tot = 0.0
        model.train()
        for b in range(0, n, batch):
            xb = X[idx[b:b + batch]]
            y, mu, lv = model(xb)
            rec = ((y - xb) ** 2).mean()
            kl = -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp())
            loss = rec + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        print(f"    VAE epoch {ep + 1}/{epochs}  loss {tot / n:.4f}")
    model.eval()
    return model


@torch.no_grad()
def vae_track_error(model, speed, turn):
    """Mean reconstruction error over a track's windows (higher = off-manifold)."""
    w = windows_from_seq(speed, turn)
    xb = torch.from_numpy(w).to(DEVICE)
    return float(model.recon_error(xb).mean().item())


# ----------------------------------------------------------------- Latent SDE
# A self-contained latent neural SDE (no torchsde): amortised encoder -> q(z0),
# a learned prior drift f_theta and posterior drift f_phi sharing one diagonal
# diffusion g, rolled forward with Euler-Maruyama over the ACTUAL per-step dt.
# The path-KL is the Girsanov rate 0.5*||(f_phi-f_theta)/g||^2 dt. Because dt
# enters the Euler step, a missed radar scan (dt = 2,3,... scan units) is handled
# natively -- the point of using an SDE over the plain VAE.

def _norm_st(speed, turn):
    s = np.asarray(speed, float) / SPEED_SCALE
    t = np.clip(np.asarray(turn, float) / TURN_SCALE, -3, 3)
    return np.column_stack([s, t]).astype(np.float32)


def windows_with_dt(speed, turn, scan=None, L=WIN, stride=1):
    """Fixed-L [speed,turn] windows plus the per-step dt (scan units). If `scan`
    is given, dt comes from the detected-scan gaps (so misses widen dt)."""
    x = _norm_st(speed, turn)
    if scan is None:
        dt = np.ones(len(x), np.float32)
    else:
        gaps = np.maximum(np.diff(np.asarray(scan)), 1).astype(np.float32)
        dt = np.concatenate([[1.0], gaps])
    if len(x) < L:
        x = np.vstack([x, np.repeat(x[-1:], L - len(x), axis=0)])
        dt = np.concatenate([dt, np.ones(L - len(dt), np.float32)])
    Xs, Ds = [], []
    for i in range(0, len(x) - L + 1, stride):
        Xs.append(x[i:i + L]); Ds.append(dt[i:i + L])
    return np.stack(Xs).astype(np.float32), np.stack(Ds).astype(np.float32)


def build_sde_training_windows(seqs, L=WIN, per_traj=6, seed=0):
    """Windows from real motion WITH random gap augmentation: pick L sorted
    indices out of a longer block, so dt in {1,2,3,...} teaches variable-rate
    dynamics. Returns X[N,L,2], DT[N,L]."""
    rng = np.random.default_rng(seed)
    Xs, Ds = [], []
    for sp, tr in seqs:
        x = _norm_st(sp, tr)
        if len(x) < L:
            continue
        for _ in range(per_traj):
            span = min(len(x), rng.integers(L, 2 * L + 1))
            start = rng.integers(0, len(x) - span + 1)
            idx = np.sort(rng.choice(np.arange(start, start + span), L, replace=False))
            Xs.append(x[idx])
            Ds.append(np.concatenate([[1.0], np.diff(idx).astype(np.float32)]))
    return np.stack(Xs).astype(np.float32), np.stack(Ds).astype(np.float32)


class LatentSDE(nn.Module):
    def __init__(self, obs=2, z=4, hidden=32):
        super().__init__()
        self.enc = nn.GRU(obs + 1, hidden, batch_first=True)   # +1 for dt channel
        self.mu0 = nn.Linear(hidden, z)
        self.lv0 = nn.Linear(hidden, z)
        self.f_prior = nn.Sequential(nn.Linear(z, hidden), nn.Tanh(), nn.Linear(hidden, z))
        self.f_post = nn.Sequential(nn.Linear(z + hidden, hidden), nn.Tanh(), nn.Linear(hidden, z))
        self.log_g = nn.Parameter(torch.zeros(z) - 1.0)        # diffusion (softplus)
        self.emit = nn.Sequential(nn.Linear(z, hidden), nn.Tanh(), nn.Linear(hidden, obs))
        self.z = z

    def _g(self):
        return torch.nn.functional.softplus(self.log_g) + 0.05

    def forward(self, X, DT):
        B, L, _ = X.shape
        _, h = self.enc(torch.cat([X, DT.unsqueeze(-1)], -1))
        ctx = h[-1]
        mu0, lv0 = self.mu0(ctx), self.lv0(ctx)
        z = mu0 + torch.randn_like(mu0) * torch.exp(0.5 * lv0)
        g = self._g()
        recon = torch.zeros(B, device=X.device)
        pathkl = torch.zeros(B, device=X.device)
        for k in range(L):
            xh = self.emit(z)
            recon = recon + ((xh - X[:, k]) ** 2).mean(-1)
            if k < L - 1:
                dt = DT[:, k + 1].clamp(min=1.0)
                fp = self.f_post(torch.cat([z, ctx], -1))
                fpr = self.f_prior(z)
                pathkl = pathkl + 0.5 * (((fp - fpr) / g) ** 2).sum(-1) * dt
                z = z + fp * dt.unsqueeze(-1) + g * torch.sqrt(dt).unsqueeze(-1) * torch.randn_like(z)
        recon = recon / L
        kl0 = -0.5 * (1 + lv0 - mu0 ** 2 - lv0.exp()).sum(-1)
        return recon, kl0, pathkl


def train_latent_sde(X, DT, epochs=12, batch=1024, seed=0, beta=0.3):
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X).to(DEVICE); Dt = torch.from_numpy(DT).to(DEVICE)
    model = LatentSDE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = len(Xt); idx = np.arange(n); rng = np.random.default_rng(seed)
    for ep in range(epochs):
        rng.shuffle(idx); tot = 0.0
        warm = min(1.0, (ep + 1) / 5) * beta          # KL warm-up for stability
        model.train()
        for b in range(0, n, batch):
            bi = idx[b:b + batch]
            rec, kl0, pk = model(Xt[bi], Dt[bi])
            loss = rec.mean() + warm * (0.01 * kl0.mean() + pk.mean())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += loss.item() * len(bi)
        print(f"    SDE epoch {ep + 1}/{epochs}  loss {tot / n:.4f}")
    model.eval()
    return model


@torch.no_grad()
def sde_track_error(model, speed, turn, scan, reps=4):
    """Mean reconstruction error over a track's windows (averaged over a few
    Euler-Maruyama rollouts to tame sampling noise). Higher = off-manifold."""
    Xw, Dw = windows_with_dt(speed, turn, scan=scan)
    xb = torch.from_numpy(Xw).to(DEVICE); db = torch.from_numpy(Dw).to(DEVICE)
    acc = 0.0
    for _ in range(reps):
        rec, _, _ = model(xb, db)
        acc += rec.mean().item()
    return acc / reps
