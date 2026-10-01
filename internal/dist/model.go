package dist

import (
	"math"

	"sampling-svc/internal/plan"
)

// Model is a defect-count distribution at a fixed defective fraction p.
//
// Probabilities are returned in the log domain (LogPmf) so callers can
// accumulate them without the tiny terms underflowing; Exp is taken only
// at the end via logSumExp. Support() gives the feasible count range and
// may collapse to a single point at p=0 or p=1.
type Model interface {
	Dist() plan.Distribution
	N() int // effective sample size
	Support() (int, int)
	LogPmf(k int) float64

	// Pmf / Cdf are exp(log-domain) conveniences kept exact at 0 and 1.
	Pmf(k int) float64
	Cdf(k int) float64
}

// NewModel builds the first-sample distribution for sample size n.
// With hypergeometric, lot is the full lot size; with binomial/poisson
// lot is ignored.
func NewModel(d plan.Distribution, n int, lot *int, p float64) Model {
	switch d {
	case plan.DistHypergeometric:
		return newHypergeo(n, *lot, p)
	case plan.DistPoisson:
		return poisson{lambda: float64(n) * p, n: n}
	default:
		return binomial{n: n, p: p}
	}
}

// Model2 is the second-sample distribution of a double plan. For
// binomial/poisson it is independent of the first count; for
// hypergeometric it is the conditional distribution given d1 defects in
// the first sample (sampling without replacement from the same lot).
type Model2 interface {
	Model
	// GivenFirst fixes the conditional parameters for hypergeometric;
	// it is a no-op for independent distributions.
	GivenFirst(d1 int)
}

// NewModel2 builds the second-stage distribution.
func NewModel2(d plan.Distribution, n1, n2 int, lot *int, p float64) Model2 {
	switch d {
	case plan.DistHypergeometric:
		h := newHypergeo(n2, *lot, p)
		return &hypergeo2{h: h, lot: *lot, n1: n1}
	case plan.DistPoisson:
		return poisson{lambda: float64(n2) * p, n: n2}
	default:
		return binomial2{binomial{n: n2, p: p}}
	}
}

// CdfLog returns log P(X <= k).
func CdfLog(m Model, k int) float64 {
	lo, hi := m.Support()
	if k < lo {
		return math.Inf(-1)
	}
	if k >= hi {
		return 0
	}
	terms := make([]float64, 0, k-lo+1)
	for x := lo; x <= k; x++ {
		terms = append(terms, m.LogPmf(x))
	}
	return logSumExp(terms)
}

// RangeLog returns log P(lo <= X <= hi).
func RangeLog(m Model, lo, hi int) float64 {
	slo, shi := m.Support()
	if lo > hi || hi < slo || lo > shi {
		return math.Inf(-1)
	}
	if lo < slo {
		lo = slo
	}
	if hi > shi {
		hi = shi
	}
	terms := make([]float64, 0, hi-lo+1)
	for x := lo; x <= hi; x++ {
		terms = append(terms, m.LogPmf(x))
	}
	return logSumExp(terms)
}

// ---------- binomial ----------

type binomial struct {
	n int
	p float64
}

func (b binomial) Dist() plan.Distribution { return plan.DistBinomial }
func (b binomial) N() int                  { return b.n }

func (b binomial) Support() (int, int) {
	switch {
	case b.p == 0:
		return 0, 0
	case b.p == 1:
		return b.n, b.n
	default:
		return 0, b.n
	}
}

func (b binomial) LogPmf(k int) float64 {
	if k < 0 || k > b.n {
		return math.Inf(-1)
	}
	switch {
	case b.p == 0:
		if k == 0 {
			return 0
		}
		return math.Inf(-1)
	case b.p == 1:
		if k == b.n {
			return 0
		}
		return math.Inf(-1)
	}
	return LogComb(b.n, k) +
		float64(k)*math.Log(b.p) +
		float64(b.n-k)*math.Log1p(-b.p)
}

func (b binomial) Pmf(k int) float64 { return math.Exp(b.LogPmf(k)) }
func (b binomial) Cdf(k int) float64 {
	l := CdfLog(b, k)
	if math.IsInf(l, -1) {
		return 0
	}
	if l == 0 {
		return 1
	}
	return math.Exp(l)
}

// binomial2 stage is an independent binomial.
type binomial2 struct{ binomial }

func (binomial2) GivenFirst(int) {}

// ---------- poisson ----------

type poisson struct {
	lambda float64
	n      int // sample size; only used for clipping support
}

func (poisson) Dist() plan.Distribution { return plan.DistPoisson }
func (x poisson) N() int                { return x.n }

func (x poisson) Support() (int, int) {
	switch {
	case x.lambda == 0:
		return 0, 0
	default:
		// The poisson model is an approximation to the binomial for small
		// p; at p=1 every sampled unit is defective, so the endpoint is
		// collapsed exactly like binomial/hypergeometric (Pa=0 for c<n).
		return 0, x.n
	}
}

func (x poisson) LogPmf(k int) float64 {
	if k < 0 || k > x.n {
		return math.Inf(-1)
	}
	// Exact endpoints match the underlying binomial even though poisson
	// is an approximation on the interior.
	if x.n > 0 && x.lambda == float64(x.n) {
		if k == x.n {
			return 0
		}
		return math.Inf(-1)
	}
	if x.lambda == 0 {
		if k == 0 {
			return 0
		}
		return math.Inf(-1)
	}
	lg, _ := math.Lgamma(float64(k) + 1)
	return float64(k)*math.Log(x.lambda) - x.lambda - lg
}

func (x poisson) Pmf(k int) float64 { return math.Exp(x.LogPmf(k)) }
func (x poisson) Cdf(k int) float64 {
	l := CdfLog(x, k)
	if math.IsInf(l, -1) {
		return 0
	}
	if l == 0 {
		return 1
	}
	return math.Exp(l)
}

// GivenFirst is a no-op: poisson stages are independent.
func (poisson) GivenFirst(int) {}

// ---------- hypergeometric ----------

// hypergeo models K defects in a lot of size L, sample n:
// P(X=k)=C(K,k) C(L-K,n-k) / C(L,n), K = round(L*p), clipped to [0,L].
type hypergeo struct {
	n, lot, K int
	p         float64
}

func newHypergeo(n, lot int, p float64) *hypergeo {
	K := int(math.Floor(float64(lot)*p + 0.5))
	if K < 0 {
		K = 0
	}
	if K > lot {
		K = lot
	}
	return &hypergeo{n: n, lot: lot, K: K, p: p}
}

func (*hypergeo) Dist() plan.Distribution { return plan.DistHypergeometric }
func (h *hypergeo) N() int                { return h.n }

func (h *hypergeo) Support() (int, int) {
	lo := h.n - (h.lot - h.K)
	if lo < 0 {
		lo = 0
	}
	hi := h.K
	if hi > h.n {
		hi = h.n
	}
	if lo > hi { // theoretically impossible configuration
		return 0, -1
	}
	return lo, hi
}

func (h *hypergeo) LogPmf(k int) float64 {
	lo, hi := h.Support()
	if lo > hi || k < lo || k > hi {
		return math.Inf(-1)
	}
	return LogComb(h.K, k) +
		LogComb(h.lot-h.K, h.n-k) -
		LogComb(h.lot, h.n)
}

func (h *hypergeo) Pmf(k int) float64 { return math.Exp(h.LogPmf(k)) }
func (h *hypergeo) Cdf(k int) float64 {
	l := CdfLog(h, k)
	if math.IsInf(l, -1) {
		return 0
	}
	if l == 0 {
		return 1
	}
	return math.Exp(l)
}

// hypergeo2 is the conditional second stage. After d1 defects were seen
// in the first n1 draws, the remaining lot is (lot-n1) units with
// (K-d1) defects.
type hypergeo2 struct {
	h   *hypergeo
	lot int
	n1  int
}

func (x *hypergeo2) Dist() plan.Distribution { return plan.DistHypergeometric }
func (x *hypergeo2) N() int                  { return x.h.n }

func (x *hypergeo2) GivenFirst(d1 int) {
	remLot := x.lot - x.n1
	remK := x.h.K - d1
	K := remK
	if K < 0 {
		K = 0
	}
	if K > remLot {
		K = remLot
	}
	x.h.lot = remLot
	x.h.K = K
}

func (x *hypergeo2) Support() (int, int)  { return x.h.Support() }
func (x *hypergeo2) LogPmf(k int) float64 { return x.h.LogPmf(k) }
func (x *hypergeo2) Pmf(k int) float64    { return x.h.Pmf(k) }
func (x *hypergeo2) Cdf(k int) float64    { return x.h.Cdf(k) }
