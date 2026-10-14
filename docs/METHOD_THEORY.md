# Method and proof obligations for the certified risk atlas

This note states the guarantees of the **fixed-product-model** method in a form suitable for a paper appendix. It distinguishes exact-arithmetic results from the floating-point implementation. The results concern a finite discounted product MDP; they do not certify collision avoidance on every CARLA trajectory.

## 1. Model and assumptions

Let the finite state and action sets be $\mathcal S$ and $\mathcal A$. The product transition kernel $P(s'\mid s,a)$, LTL automata, labels, reward $r(s,a)$, discount $0<\gamma<1$, and initial distribution $\alpha$ are fixed during a risk-update sequence. The kernel is stochastic; the released 4×4 example is completed with a zero-reward absorbing state where necessary. A query changes only

$$
c_\theta(s,a)=c_0(s,a)+\sum_{j=1}^{d}\theta_j h_j(s,a).
$$

The code uses state costs, a special case of this notation. For the monotonicity certificate below, each $h_j\geq0$, and queries are restricted to a domain where $c_\theta\geq0$. The policy-region and primal–dual results only need finite costs. Let $0\leq S<H$ be the soft and hard risk thresholds. Assume the queried CMDP is feasible; otherwise the method must report infeasibility or failure.

For a discounted state-action occupation $x$, define

$$
\mathcal X_\alpha=\{x\geq0:\sum_a x(s,a)-\gamma\sum_{u,a}P(s\mid u,a)x(u,a)=\alpha(s),\ \forall s\},
\quad R(x)=\langle r,x\rangle,\quad C_\theta(x)=\langle c_\theta,x\rangle.
$$

The optimization problem inherited from the risk-LTL model is

$$
F^*(\theta)=\max_{x\in\mathcal X_\alpha,\ C_\theta(x)\leq H}
\left\{R(x)-(C_\theta(x)-S)_+\right\},\qquad (y)_+=\max(0,y). \tag{1}
$$

All statements below use **expected discounted** reward and risk. Changing $P$, labels, automata, or reward is outside their scope. When $\alpha$ changes, occupation measures and scalarized values must be recomputed for that initial distribution; the Bellman policy regions themselves remain valid.

## 2. Why a cached policy has a reusable region

For a deterministic stationary policy $\pi$, let $P_\pi$ be its transition matrix. At a risk price $t\geq0$, the scalarized ordinary MDP has reward $r-tc_\theta$. Introduce the joint coordinates

$$
z=(z_0,z_1,\ldots,z_d)=(t,t\theta_1,\ldots,t\theta_d).
$$

The scalarized value of $\pi$ is

$$
v_\pi(z)=(I-\gamma P_\pi)^{-1}
\left(r_\pi-z_0c_{0,\pi}-\sum_jz_jh_{j,\pi}\right). \tag{2}
$$

The inverse exists because $\gamma<1$ and $P_\pi$ is stochastic. Thus $v_\pi$ is affine in $z$. Define its Bellman region by all action inequalities

$$
\mathcal D_\pi=\left\{z:z_0\geq0,\quad
v_\pi(z)(s)\geq r(s,a)-z_0c_0(s,a)-\sum_jz_jh_j(s,a)
+\gamma\sum_{s'}P(s'\mid s,a)v_\pi(z)(s')\quad\forall(s,a)\right\}. \tag{3}
$$

**Proposition 1 (region validity).** $\mathcal D_\pi$ is a polyhedron. If the query ray $z=t(1,\theta)$ intersects $\mathcal D_\pi$, then $\pi$ is an optimal policy for the scalarized MDP at that $(t,\theta)$. Ties with other policies are allowed.

**Proof.** Each inequality in (3) is affine in $z$ by (2), hence their intersection is a polyhedron. The value in (2) satisfies the Bellman **equality** for policy $\pi$. Inequalities (3) say no alternative action has a larger Bellman value. Consequently $v_\pi$ is a fixed point of the optimal Bellman operator. This operator is a $\gamma$-contraction, so its fixed point is unique and equals the scalarized optimum. □

For fixed $\theta$, substituting $z=t(1,\theta)$ into (3) gives an interval of valid nonnegative prices. The code stores affine inequality coefficients in `joint_price_atlas.py: add_policy` and computes this interval in `price_interval`. This is a policy-*optimality* region for the scalarized MDP, **not** a claim that the policy alone solves the constrained problem (1).

## 3. Feasible cached policies and their mixtures

**Lemma 2 (occupation-mixture realization).** Suppose $x_1,\ldots,x_m\in\mathcal X_\alpha$. Any convex combination $x=\sum_i w_i x_i$, with $w_i\geq0$ and $\sum_iw_i=1$, also belongs to $\mathcal X_\alpha$. It is the occupation of a stationary randomized policy

$$
\pi_x(a\mid s)=\frac{x(s,a)}{\sum_bx(s,b)}
$$

when the denominator is positive; actions at zero-occupation states may be chosen arbitrarily.

**Proof.** Nonnegativity and the flow equations defining $\mathcal X_\alpha$ are linear, so they hold for $x$. Writing $d(s)=\sum_a x(s,a)$ and substituting $\pi_x$ into the flow equation gives $d=\alpha+\gamma P_{\pi_x}^{\mathsf T}d$. Since $I-\gamma P_{\pi_x}^{\mathsf T}$ is invertible, this is the unique discounted state occupation of $\pi_x$. Therefore its state-action occupation is $x$. □

Consequently, a cached combination with $C_\theta(x)\leq H$ gives an attainable lower bound

$$
L(\theta)=R(x)-(C_\theta(x)-S)_+\leq F^*(\theta). \tag{4}
$$

**Proposition 3 (two cached policies suffice for the cache hull).** The maximum of (1) **restricted to the convex hull of cached feasible-flow occupations** is attained either by one cached occupation or by a mixture of two whose risk equals $S$ or $H$. This does not say that two *globally optimal* policies always suffice without searching beyond the cache.

**Proof.** Write a hull point as weights $w$ in the simplex. On $C_\theta\leq S$, the objective is the linear function $R$; on $S\leq C_\theta\leq H$, it is the linear function $R-C_\theta+S$. An optimum of either linear program is attained at an extreme point. A simplex cut by a single active scalar-risk boundary has an extreme point with at most two nonzero weights. The only possible boundary values are $S$ and $H$, because $S<H$. Interior extreme points are original simplex vertices. Enumerating singletons and two-policy intersections with $S,H$ therefore attains the hull optimum. □

`joint_price_atlas.query` and `local_residual.cached_primal` enumerate these candidates. The implementation now computes risk from the **actual clipped mixing weight**, then evaluates the hinge penalty at that risk. If the initial state changes, `rebase_initial` uses each cached $(I-\gamma P_\pi)$ factorization to regenerate its occupation before applying this lemma.

## 4. A global upper bound from any risk price

Let $V_t^\star(\theta)$ be the optimal value function for reward $r-tc_\theta$, without the hard risk constraint. Define

$$
b(t)=S\min(1,t)+H(t-1)_+.
$$

**Theorem 4 (primal–dual certificate).** For every feasible query of (1) and every $t\geq0$,

$$
F^*(\theta)\leq \alpha^{\mathsf T}V_t^\star(\theta)+b(t). \tag{5}
$$

More generally, for **any** candidate scalarized value vector $v$, define its one-sided Bellman residual

$$
\delta(v,t,\theta)=\max_{s,a}\bigl[r(s,a)-tc_\theta(s,a)
+\gamma\sum_{s'}P(s'\mid s,a)v(s')-v(s)\bigr]_+.
$$

Then a valid upper bound is

$$
U(\theta;t,v)=\alpha^{\mathsf T}v+
\frac{\delta(v,t,\theta)}{1-\gamma}+b(t). \tag{6}
$$

**Proof.** If $0\leq t\leq1$, $(C-S)_+\geq t(C-S)$, so every feasible $x$ has
$R(x)-(C-S)_+\leq R(x)-tC+tS$. If $t\geq1$, then $(C-S)_+\geq C-S$, and $C\leq H$ implies
$R(x)-(C-S)_+\leq R(x)-tC+S+(t-1)H$. Maximizing the unconstrained scalarized reward yields (5). For (6), write $T_t$ for the optimal Bellman operator. By definition $T_tv\leq v+\delta\mathbf1$. Since rows of $P$ sum to one,
$T_t(v+\delta\mathbf1/(1-\gamma))\leq v+\delta\mathbf1/(1-\gamma)$.
Monotonicity and contraction imply $V_t^\star\leq v+\delta\mathbf1/(1-\gamma)$. Substitute this in (5). □

Inside a cached region, the residual is zero in exact arithmetic. `joint_price_atlas.dual_bound` explicitly recomputes it at the selected price and adds the correction in (6); the region lookup is therefore a speed heuristic, not the sole source of upper-bound validity. The code also applies a small floating-point margin.

**Corollary 5 (safe return condition).** If a cached occupation or mixture yields $C_\theta(x)\leq H$, and (6) yields $U-L\leq\varepsilon$, then

$$
0\leq F^*(\theta)-L\leq\varepsilon. \tag{7}
$$

**Proof.** Combine (4), $L\leq F^*$, with (6), $F^*\leq U$. □

The implementation uses $\varepsilon=\max(10^{-4},0.01|L|)$ plus numerical tolerances. This is an **absolute-or-relative** criterion, not a strict 1% ratio when $L$ is near zero.

### Monotone upper-bound shortcut used by the hybrid method

**Corollary 6.** If all $h_j\geq0$, the initial distribution is unchanged, and $\theta'\geq\theta$ componentwise, then $F^*(\theta')\leq F^*(\theta)$. Hence any valid upper bound stored at $\theta$ remains an upper bound at $\theta'$.

**Proof.** For every occupation $x\geq0$, $C_{\theta'}(x)\geq C_\theta(x)$. The feasible set at $\theta'$ is a subset of the one at $\theta$, while the hinge penalty cannot decrease. Taking maxima gives the claim. □

This shortcut appears in `stream_certificate.query`. The CARLA closed-loop orchestrator clears start-dependent upper-bound archives when the initial product state changes. If directions have mixed signs, or transitions/labels change, this shortcut is unavailable.

For changes in either direction, let $B_j$ be an upper bound on
$\max_{x\in\mathcal X_\alpha}\langle h_j,x\rangle$; the code obtains such
bounds by Bellman value iteration from an initial supersolution. Suppose an
archive entry at $\theta$ stores a **price-specific** upper bound
$U_t(\theta)=\alpha^{\mathsf T}V_t^\star(\theta)+b(t)$ (or a padded upper
bound for the same $t$). Then, for $h_j\geq0$,

$$
F^*(\theta')\leq U_t(\theta)+t\sum_j B_j(\theta_j-\theta'_j)_+.
\tag{8}
$$

Indeed, for every occupation $x$, the change in its scalarized reward is
$t\sum_j(\theta_j-\theta'_j)\langle h_j,x\rangle$, which is no larger than
the added term in (8). Maximizing over the same flow polytope and applying
(5) proves the bound. This is the `directional_reach` certificate. An
arbitrary upper bound on $F^*(\theta)$ cannot be paired with an arbitrary
price in (8): the archived price and upper bound must come from the **same
scalarized dual**. For an exact LP anchor, the combined soft and hard risk
multipliers supply that price by strong LP duality. All these bounds depend
on $\alpha$, so a changed start requires a fresh archive.

## 5. Query, expansion and the limits of the fallback

The observed-initialization method first solves (1) at the first observed $\theta_0$, storing scalarized deterministic policies, occupations and Bellman-region coefficients. For each later query it:

1. forms the best feasible cached single policy or two-policy mixture;
2. tries a valid monotone or risk-price upper bound and returns only if (7) holds;
3. on failure, solves one ordinary scalarized MDP at a nearby price, adds any new policy, and checks again;
4. if necessary, runs the complete CMDP price search, adds its oracle policies, and checks again;
5. reports failure rather than using an uncertified candidate if the finite search budget cannot establish (7).

**Proposition 7 (fail-closed output).** Under exact arithmetic and the assumptions of Section 1, every query returned with `certified=True` by the atlas/hybrid path is feasible for (1) and meets (7). Adding policies cannot invalidate an existing primal or dual certificate. The finite fallback does **not** by itself guarantee that every feasible query will be certified: a bounded price search or numerical solver may fail, in which case no certified strategy may be returned.

**Proof.** Steps 1–4 only enlarge the set of cached flow-feasible occupations and upper-bound candidates. The return gate checks hard-risk feasibility and the lower/upper gap. Lemma 2 and Corollary 5 then give feasibility and (7), independently of which step generated the cached policies. A failed check triggers further search or a failure report. The finite search budget gives no completeness theorem. □

`paper_atlas_comparison.py` and the CARLA closed-loop planner explicitly raise on a failed final certificate. Thus “complete CMDP fallback” means **a more expensive final attempt followed by verification**, not permission to drive on its unverified output. Neither this proposition nor the experiments prove that the method is always faster than warm-started LP.

## 6. Numerical and application boundaries

Results 1–7 are exact-arithmetic statements about the stated product MDP. The Python implementation uses double-precision sparse linear algebra, residual checks and independent policy/LP spot checks. Its `certified` flag is a **numerically checked model certificate**, not a machine-verified interval-arithmetic proof. In particular, obtaining a strict floating-point guarantee would require outward-rounded bounds for value/flow residuals and for the reported risk and objective.

CARLA supplies observations and executes a controller coupled to a high-level grid model. The guarantee in (7) applies to that grid model's expected discounted quantities, conditional on fixed transitions and labels; it does not imply pathwise collision freedom, calibrated pedestrian probabilities or a 20 Hz worst-case planning bound. The older CARLA timing JSON predates the small residual-accounting correction in this note; it has **not** been rerun with the revised code, so it must not be quoted as new-code CARLA timing.

## 7. Checks of the revised implementation

`check_joint_atlas.py` passed 300 archived main-experiment point comparisons, 100 changed-start comparisons and 25 freshly solved large-update points against independent LP results. `docs/validation/check_certificate_math.py` independently compares the residual upper bound with scalarized MDP optima and the two-policy enumeration with a hull LP. `test_directional_reach.py` checked (8) at 40 parameter changes against independent LP optima. The following focused paired runs used the revised code, the same fixed product and 20 risk queries per case. Each method ran in a separate process; times include its own preparation and online queries, but exclude common product construction and independent post-run checks.

| Case | Hybrid atlas | Warm-started LP | Direct atlas hits | Queries meeting the hard-risk and 1% checks |
| --- | ---: | ---: | ---: | ---: |
| 24×24, local `walk` | 2.208 s | 10.612 s | 16/20 | 20/20 |
| 32×32, local `walk` | 4.305 s | 38.201 s | 16/20 | 20/20 |
| 16×16, active hard budget, `iid` | 6.360 s | 5.421 s | 3/20 | 20/20 |

The hard budget is active in 16/20 queries of the last case, and the atlas is slower there. These are three single-run checks, not a claim of uniform speed superiority or a statistically stable timing estimate. The detailed JSON, including per-query policy-evaluation errors, is in `docs/validation/paired_20261014/` and `docs/validation/paired_hard_20261014/`. The original CARLA closed-loop run was not repeated for this proof update.

From the repository root, reproduce the mathematical checks and the three paired planning runs with:

```bash
python docs/validation/check_certificate_math.py
python reproduction/adaptive_risk_ltl/check_joint_atlas.py
python reproduction/adaptive_risk_ltl/test_directional_reach.py
python reproduction/adaptive_risk_ltl/run_paper_atlas_suite.py --select 0 2 --methods warm_lp hybrid_atlas --count 20 --output-dir docs/validation/paired_20261014
python reproduction/adaptive_risk_ltl/run_paper_atlas_suite.py --select 6 --methods warm_lp hybrid_atlas --count 20 --output-dir docs/validation/paired_hard_20261014
```

Wall times depend on the machine and installed solver versions. The committed JSON records the run reported above.

Implementation: `reproduction/adaptive_risk_ltl/joint_price_atlas.py`, `local_residual.py`, `stream_certificate.py`, `stratified.py`, and `paper_atlas_comparison.py`. Experimental evidence and counterexamples: `JOINT_ATLAS_REPORT.md` and `reproduction/carla_local_risk/observed_hybrid_probe/FOUR_CASE_AND_ARRIVAL_CN.md`.
