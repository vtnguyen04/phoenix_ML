package logger

type Logger interface {
    Error(msg string, keysAndValues ...interface{})
    Warn(msg string, keysAndValues ...interface{})
}
