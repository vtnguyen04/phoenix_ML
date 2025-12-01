package metrics

// MetricsClient is an interface for a metrics client.
type MetricsClient interface {
	IncrementCounter(name string)
	RecordHistogram(name string, value float64)
}
