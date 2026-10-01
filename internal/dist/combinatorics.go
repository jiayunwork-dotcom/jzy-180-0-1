package dist

import (
	"math"
)

// LogComb returns log(C(n,k)). Inputs outside 0<=k<=n give -Inf, so
// impossible outcomes never silently contribute probability mass.
//
// Everything is evaluated in the log domain via the log-gamma function;
// for n up to several thousand no intermediate factorial is ever formed.
func LogComb(n, k int) float64 {
	if k < 0 || k > n || n < 0 {
		return math.Inf(-1)
	}
	if k == 0 || k == n {
		return 0
	}
	// C(n,k) == C(n,n-k): use the smaller k to keep cancellation minimal.
	if k > n-k {
		k = n - k
	}
	l, _ := math.Lgamma(float64(n) + 1)
	a, _ := math.Lgamma(float64(k) + 1)
	b, _ := math.Lgamma(float64(n-k) + 1)
	return l - a - b
}

// logSumExp sums exp(ls) without under/overflow. Empty input is -Inf.
func logSumExp(ls []float64) float64 {
	m := math.Inf(-1)
	for _, v := range ls {
		if v > m {
			m = v
		}
	}
	if math.IsInf(m, -1) {
		return m
	}
	var s float64
	for _, v := range ls {
		s += math.Exp(v - m)
	}
	return m + math.Log(s)
}
