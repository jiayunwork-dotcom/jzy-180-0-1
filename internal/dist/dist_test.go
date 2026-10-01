package dist

import (
	"math"
	"testing"

	"sampling-svc/internal/plan"
)

func approxEq(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestLogComb(t *testing.T) {
	cases := []struct {
		n, k int
		want float64
	}{
		{5, 0, 0}, {5, 5, 0}, {5, 2, math.Log(10)},
		{10, 3, math.Log(120)},
	}
	for _, tc := range cases {
		if got := LogComb(tc.n, tc.k); !approxEq(got, tc.want, 1e-12) {
			t.Fatalf("C(%d,%d)=%v want %v", tc.n, tc.k, got, tc.want)
		}
	}
	// C(100,50) = 1.008913445...e29 -> log = 66.783841...
	if got := LogComb(100, 50); !approxEq(got, 66.783841, 1e-6) {
		t.Fatalf("C(100,50) log=%f", got)
	}
	// Out of range -> -Inf (never a silent zero contribution).
	if !math.IsInf(LogComb(5, 6), -1) || !math.IsInf(LogComb(5, -1), -1) {
		t.Fatal("out-of-range LogComb must be -Inf")
	}
	// Large n must not overflow in the log domain.
	l := LogComb(5000, 2500)
	if math.IsInf(l, 0) || math.IsNaN(l) || l <= 0 {
		t.Fatalf("LogComb(5000,2500)=%v must be finite and positive", l)
	}
}

func TestBinomialEndpoints(t *testing.T) {
	b := NewModel(plan.DistBinomial, 80, nil, 0)
	if got := b.Cdf(2); got != 1 {
		t.Fatalf("p=0 CDF must be exactly 1, got %v", got)
	}
	b1 := NewModel(plan.DistBinomial, 80, nil, 1)
	if got := b1.Cdf(2); got != 0 {
		t.Fatalf("p=1, c<n CDF must be exactly 0, got %v", got)
	}
	if got := b1.Cdf(80); got != 1 {
		t.Fatalf("p=1, c=n CDF must be 1, got %v", got)
	}
}

func TestBinomialSumsToOne(t *testing.T) {
	for _, p := range []float64{0.01, 0.3, 0.5, 0.9} {
		m := NewModel(plan.DistBinomial, 60, nil, p)
		var s float64
		for k := 0; k <= 60; k++ {
			s += m.Pmf(k)
		}
		if !approxEq(s, 1, 1e-12) {
			t.Fatalf("binomial pmf sum at p=%v is %v", p, s)
		}
	}
}

func TestPoissonSumsAndClip(t *testing.T) {
	m := NewModel(plan.DistPoisson, 50, nil, 0.04)
	var s float64
	for k := 0; k <= 50; k++ {
		s += m.Pmf(k)
	}
	if !approxEq(s, 1, 1e-6) {
		t.Fatalf("poisson clipped sum=%v", s)
	}
}

func TestHypergeometricSumsAndEndpoint(t *testing.T) {
	N := 500
	m := NewModel(plan.DistHypergeometric, 50, &N, 0.1)
	var s float64
	lo, hi := m.Support()
	for k := lo; k <= hi; k++ {
		s += m.Pmf(k)
	}
	if !approxEq(s, 1, 1e-12) {
		t.Fatalf("hypergeometric pmf sum=%v support=[%d,%d]", s, lo, hi)
	}
	// p=0 -> all counts at zero; p=1 -> all sampled units defective.
	m0 := NewModel(plan.DistHypergeometric, 50, &N, 0)
	if m0.Cdf(2) != 1 {
		t.Fatal("hyper p=0 Pa must be exactly 1")
	}
	m1 := NewModel(plan.DistHypergeometric, 50, &N, 1)
	if m1.Cdf(2) != 0 {
		t.Fatal("hyper p=1 c<n Pa must be exactly 0")
	}
}

// As N >> n, hypergeometric Pa approaches binomial Pa. The finite-
// population correction is ~ (n/N)*p(1-p); verify monotone convergence
// and agreement to four digits once N is 1250x the sample size.
func TestHyperApproachesBinomial(t *testing.T) {
	n, c := 80, 2
	for _, p := range []float64{0.01, 0.05, 0.1} {
		bm := NewModel(plan.DistBinomial, n, nil, p)
		bin := bm.Cdf(c)
		prevGap := math.Inf(1)
		for _, N := range []int{1000, 4000, 16000, 100000} {
			hm := NewModel(plan.DistHypergeometric, n, &N, p)
			gap := math.Abs(hm.Cdf(c) - bin)
			if gap > prevGap+1e-12 {
				t.Fatalf("p=%v N=%d gap %v did not shrink from %v", p, N, gap, prevGap)
			}
			prevGap = gap
		}
		N := 100000
		hm := NewModel(plan.DistHypergeometric, n, &N, p)
		if math.Abs(hm.Cdf(c)-bin) > 1e-3 {
			t.Fatalf("large N mismatch p=%v gap=%v", p, math.Abs(hm.Cdf(c)-bin))
		}
	}
}
