package qupath.ext.wsinsight.runner;

/**
 * Callback interface for streaming Docker process output and lifecycle events
 * to a UI progress dialog or other consumer.
 */
public interface ProgressListener {

    /** Called for each stdout/stderr line from the container. */
    void onLogLine(String line);

    /** Called when the job finishes (successfully or not). */
    void onFinished(int exitCode);

    /** Called when an exception bubbles out of the launcher. */
    void onError(Throwable t);
}
