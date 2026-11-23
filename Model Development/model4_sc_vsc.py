"""StintIQ - Model 4 runtime: SC/VSC discrete-time hazard predictor (imported by Model 3 / Model 5)."""
import numpy as np
import pandas as pd
import joblib


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class ConstantHazard:
    """Climatology baseline: one constant per-lap hazard."""
    def __init__(self):
        self.p_ = 0.0

    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p_), np.full(n, self.p_)])


class Model4Predictor:
    """Calibrated discrete-time hazard h_t = P(new SC/VSC starts on lap t | none active, state at start of t).

    ctx keys used as scenario inputs (all optional except total_laps):
        total_laps, street_level, wall_risk, hist_sc_rate, rain_prev, wet_frac_prev,
        track_temp, air_temp, humidity, rain_change
    Incident / field features that are unknown in the future are held at their 'quiet-state' training medians.
    """

    def __init__(self, model, boot_models, features_all, shift, quiet,
                 dur_sc, dur_vsc, n_sc, n_vsc, pace_ratio=None, meta=None):
        self.model = model
        self.boot = list(boot_models or [])
        self.features_all = list(features_all)
        self.shift = float(shift)
        self.quiet = dict(quiet)
        self.dur_sc = np.asarray(dur_sc, dtype=int)
        self.dur_vsc = np.asarray(dur_vsc, dtype=int)
        self.n_sc, self.n_vsc = int(n_sc), int(n_vsc)
        self.pace_ratio = dict(pace_ratio or {})
        self.meta = dict(meta or {})

    # ------------------------------------------------------------------ core prediction
    def _predict(self, model, X):
        p = model.predict_proba(X)[:, 1]
        return _sigmoid(_logit(p) + self.shift)

    def hazard(self, X, with_band=False):
        p = self._predict(self.model, X)
        if not with_band or not self.boot:
            return p
        P = np.column_stack([self._predict(m, X) for m in self.boot])
        return p, np.percentile(P, 10, axis=1), np.percentile(P, 90, axis=1)

    def _make_X(self, ctx, t, since, n_prior):
        t = np.asarray(t, dtype=float)
        k = len(t)
        T = float(ctx['total_laps'])
        d = {c: np.full(k, self.quiet.get(c, np.nan), dtype=float) for c in self.features_all}
        for c in self.features_all:
            if c in ctx:
                d[c] = np.full(k, float(ctx[c]))
        since = np.minimum(np.asarray(since, dtype=float), 40)
        n_prior = np.asarray(n_prior, dtype=float)
        d['progress'] = t / T
        d['laps_remaining'] = T - t
        d['is_lap1'] = (t == 1).astype(float)
        d['laps_since_neut'] = since
        d['n_prior_neut'] = n_prior
        d['restart_window'] = ((n_prior > 0) & (since <= 2)).astype(float)
        return pd.DataFrame({c: d[c] for c in self.features_all})

    # ------------------------------------------------------------------ hazard curve for Model 3
    def hazard_curve(self, ctx, from_lap=1, last_neutral_lap=0, n_prior=0):
        """Per-lap hazard (+10/90 % bootstrap band) and cumulative P(first onset <= t), assuming nothing
        happens between from_lap and t (i.e. 'first event from now')."""
        T = int(ctx['total_laps'])
        laps = np.arange(int(from_lap), T + 1)
        since = (laps - 1) - last_neutral_lap
        X = self._make_X(ctx, laps, since, np.full(len(laps), n_prior))
        p, lo, hi = self.hazard(X, with_band=True) if self.boot else (self.hazard(X),) * 3
        return pd.DataFrame({
            'lap': laps, 'hazard': p, 'hazard_lo': lo, 'hazard_hi': hi,
            'cum_prob': 1 - np.cumprod(1 - p),
            'cum_lo': 1 - np.cumprod(1 - lo), 'cum_hi': 1 - np.cumprod(1 - hi)})

    @staticmethod
    def prob_within(curve, start_lap, horizon):
        """P(at least one onset in laps [start_lap, start_lap+horizon-1]) from a hazard_curve table."""
        m = (curve['lap'] >= start_lap) & (curve['lap'] < start_lap + horizon)
        return float(1 - np.prod(1 - curve.loc[m, 'hazard'].values)) if m.any() else np.nan

    # ------------------------------------------------------------------ Monte Carlo sampler for Model 3
    def _sample_dur(self, rng, kind, k):
        arr = self.dur_sc if kind == 'SC' else self.dur_vsc
        if len(arr) < 5:
            pooled = np.concatenate([self.dur_sc, self.dur_vsc])
            arr = pooled if len(pooled) >= 5 else np.array([4] if kind == 'SC' else [2])
        return rng.choice(arr, size=k)

    def simulate(self, ctx, n_sims=2000, rng=None, from_lap=1, last_neutral_lap=0, n_prior=0):
        """State-aware simulation of neutralisations. Parameter uncertainty is propagated: each simulated race
        uses one bootstrap hazard model and its own draw of P(SC | onset) ~ Beta posterior."""
        rng = rng or np.random.default_rng(0)
        T, n = int(ctx['total_laps']), int(n_sims)
        models = self.boot[:20] if self.boot else [self.model]
        G = len(models)
        gid = np.arange(n) % G
        p_sc = rng.beta(2 + self.n_sc, 2 + self.n_vsc, size=n)
        left = np.zeros(n, int)
        prev = np.zeros(n, bool)
        last_neut = np.full(n, last_neutral_lap)
        n_pri = np.full(n, n_prior)
        neutral = np.zeros((n, T + 1), bool)
        events = []
        for t in range(int(from_lap), T + 1):
            ongoing = left > 0
            risk = (~ongoing) & (~prev)
            h = np.zeros(n)
            for g in range(G):
                m = risk & (gid == g)
                if m.any():
                    X = self._make_X(ctx, np.full(m.sum(), t), (t - 1) - last_neut[m], n_pri[m])
                    h[m] = self._predict(models[g], X)
            new = risk & (rng.random(n) < h)
            if new.any():
                idx = np.where(new)[0]
                is_sc = rng.random(len(idx)) < p_sc[idx]
                dur = np.where(is_sc, self._sample_dur(rng, 'SC', len(idx)), self._sample_dur(rng, 'VSC', len(idx)))
                left[idx] = np.maximum(dur, 1)
                n_pri[idx] += 1
                events.extend((int(i), t, 'SC' if s else 'VSC', int(du)) for i, s, du in zip(idx, is_sc, dur))
            cur = left > 0
            neutral[cur, t] = True
            left[cur] -= 1
            last_neut[cur] = t
            prev = cur
        ev = pd.DataFrame(events, columns=['sim', 'lap', 'type', 'duration'])
        return {'events': ev, 'neutral': neutral[:, 1:], 'n_sims': n}

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
