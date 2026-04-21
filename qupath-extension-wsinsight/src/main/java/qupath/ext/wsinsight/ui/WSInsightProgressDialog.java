package qupath.ext.wsinsight.ui;

import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.TextArea;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;

import qupath.ext.wsinsight.runner.DockerRunner;
import qupath.ext.wsinsight.runner.ProgressListener;

/**
 * Modal JavaFX dialog that launches a {@link DockerRunner} on a background
 * thread and streams container output into a scrolling text area.
 */
public class WSInsightProgressDialog {

    private final Stage stage;
    private final TextArea logArea;
    private final ProgressBar progressBar;
    private final Label statusLabel;
    private final Button cancelButton;
    private final DockerRunner runner;

    private volatile Integer exitCode;
    private Runnable onFinished;

    public WSInsightProgressDialog(String title, DockerRunner runner) {
        this.runner = runner;
        this.stage = new Stage();
        stage.setTitle(title);
        stage.initModality(Modality.APPLICATION_MODAL);

        this.logArea = new TextArea();
        logArea.setEditable(false);
        logArea.setWrapText(false);
        logArea.setPrefColumnCount(100);
        logArea.setPrefRowCount(24);

        this.progressBar = new ProgressBar();
        progressBar.setPrefWidth(Double.MAX_VALUE);
        progressBar.setProgress(ProgressBar.INDETERMINATE_PROGRESS);

        this.statusLabel = new Label("Launching container…");
        this.cancelButton = new Button("Cancel");
        cancelButton.setOnAction(ev -> {
            cancelButton.setDisable(true);
            statusLabel.setText("Cancelling…");
            new Thread(runner::cancel, "wsinsight-docker-kill").start();
        });
        Button closeButton = new Button("Close");
        closeButton.setDisable(true);
        closeButton.setOnAction(ev -> stage.close());

        HBox buttons = new HBox(8, cancelButton, closeButton);
        VBox bottom = new VBox(6, progressBar, statusLabel, buttons);
        bottom.setPadding(new Insets(8));

        BorderPane root = new BorderPane(logArea);
        root.setBottom(bottom);
        root.setPadding(new Insets(8));
        stage.setScene(new Scene(root, 880, 560));

        // Swap Cancel for Close when the job ends.
        stage.setOnHidden(ev -> { if (onFinished != null) onFinished.run(); });
        this.cancelOnClose = closeButton;
    }

    private final Button cancelOnClose;

    public void setOnFinished(Runnable r) { this.onFinished = r; }

    /** Run the container. Blocks the calling thread only long enough to show the dialog. */
    public void showAndRun() {
        Task<Integer> task = new Task<>() {
            @Override
            protected Integer call() throws Exception {
                return runner.run(new ProgressListener() {
                    @Override public void onLogLine(String line) {
                        Platform.runLater(() -> logArea.appendText(line + "\n"));
                    }
                    @Override public void onFinished(int exitCode) {
                        // Handled after waitFor below.
                    }
                    @Override public void onError(Throwable t) {
                        Platform.runLater(() -> logArea.appendText("ERROR: " + t + "\n"));
                    }
                });
            }
        };

        task.setOnSucceeded(ev -> {
            exitCode = task.getValue();
            progressBar.setProgress(exitCode == 0 ? 1.0 : 0.0);
            statusLabel.setText(exitCode == 0
                    ? "Finished successfully."
                    : "Finished with exit code " + exitCode + ".");
            cancelButton.setDisable(true);
            cancelOnClose.setDisable(false);
        });
        task.setOnFailed(ev -> {
            Throwable t = task.getException();
            logArea.appendText("ERROR: " + (t == null ? "unknown" : t.toString()) + "\n");
            progressBar.setProgress(0.0);
            statusLabel.setText("Failed.");
            cancelButton.setDisable(true);
            cancelOnClose.setDisable(false);
            exitCode = -1;
        });

        Thread t = new Thread(task, "wsinsight-runner");
        t.setDaemon(true);
        t.start();
        stage.showAndWait();
    }

    public Integer getExitCode() { return exitCode; }
}
