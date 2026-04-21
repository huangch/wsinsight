package qupath.ext.wsinsight;

import javafx.scene.control.Menu;
import javafx.scene.control.MenuItem;
import javafx.scene.control.SeparatorMenuItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.ext.wsinsight.commands.WSInsightCommands;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.panes.PreferencePane;

/**
 * Entry point for the WSInsight QuPath extension.
 * <p>
 * Registers persistent preferences in the "WSInsight" preference category and
 * populates the {@code Extensions > WSInsight} menu with one entry per
 * WSInsight CLI subcommand. Each menu item launches the same
 * {@link qupath.ext.wsinsight.ui.WSInsightProgressDialog} after collecting
 * arguments through a generated form.
 */
public class WSInsightExtension implements QuPathExtension, GitHubProject {

    private static final Logger logger = LoggerFactory.getLogger(WSInsightExtension.class);
    private static final String MENU_NAME = "Extensions>WSInsight";
    private boolean installed;

    @Override
    public void installExtension(QuPathGUI qupath) {
        if (installed) return;
        installed = true;
        logger.info("Installing WSInsight extension v0.1.0");

        registerPreferences(qupath);
        addMenuItems(qupath);
    }

    private void registerPreferences(QuPathGUI qupath) {
        WSInsightSetup s = WSInsightSetup.getInstance();
        PreferencePane prefs = qupath.getPreferencePane();

        prefs.addPropertyPreference(s.dockerBinaryProperty(), String.class,
                "Docker binary", "WSInsight",
                "Path to the docker executable (default 'docker').");
        prefs.addPropertyPreference(s.dockerImageProperty(), String.class,
                "Docker image", "WSInsight",
                "WSInsight Docker image tag (e.g. huangchtw/wsinsight:latest).");
        prefs.addPropertyPreference(s.gpusProperty(), String.class,
                "GPUs", "WSInsight",
                "Value for docker --gpus (e.g. 'all', 'none', 'device=0', 'device=0,1').");
        prefs.addPropertyPreference(s.shmSizeProperty(), String.class,
                "Shared memory size", "WSInsight",
                "Value for docker --shm-size. Use '32g' for multi-worker dataloaders.");
        prefs.addPropertyPreference(s.hostWsiRootProperty(), String.class,
                "Host WSI root (→ /slides)", "WSInsight",
                "Host directory bound to /slides inside the container.");
        prefs.addPropertyPreference(s.hostResultsRootProperty(), String.class,
                "Host results root (→ /results)", "WSInsight",
                "Host directory bound to /results inside the container.");
        prefs.addPropertyPreference(s.extraMountsProperty(), String.class,
                "Extra mounts", "WSInsight",
                "Additional bind mounts, separated by commas/semicolons/newlines. "
                        + "Format: 'host/path:/container/path'.");
        prefs.addPropertyPreference(s.zooRegistryProperty(), String.class,
                "WSInfer Zoo registry path", "WSInsight",
                "Value passed as WSINFER_ZOO_REGISTRY_PATH inside the container.");
        prefs.addPropertyPreference(s.s3OptionsProperty(), String.class,
                "S3 storage options (JSON)", "WSInsight",
                "Value passed as S3_STORAGE_OPTIONS inside the container.");
        prefs.addPropertyPreference(s.cacheDirProperty(), String.class,
                "Remote cache directory", "WSInsight",
                "Value passed as WSINSIGHT_REMOTE_CACHE_DIR inside the container.");
        prefs.addPropertyPreference(s.kerasHomeProperty(), String.class,
                "KERAS_HOME", "WSInsight",
                "Override Keras config/weights directory inside the container.");
        prefs.addPropertyPreference(s.autoImportResultsProperty(), Boolean.class,
                "Auto-import results", "WSInsight",
                "Import GeoJSON annotations and OME-CSV measurements back into the "
                        + "active QuPath project when a job finishes successfully.");
    }

    private void addMenuItems(QuPathGUI qupath) {
        Menu menu = qupath.getMenu(MENU_NAME, true);

        menu.getItems().addAll(
                item("Run (one-shot pipeline)…", () -> WSInsightCommands.run().showAndRun()),
                new SeparatorMenuItem(),
                item("Patch…",          () -> WSInsightCommands.patch().showAndRun()),
                item("Infer…",          () -> WSInsightCommands.infer().showAndRun()),
                item("Region registration…", () -> WSInsightCommands.reg().showAndRun()),
                new SeparatorMenuItem(),
                item("H-plot…",                () -> WSInsightCommands.hplot().showAndRun()),
                item("H-plot finalize…",       () -> WSInsightCommands.hplotFinalize().showAndRun()),
                item("Neighborhood composition (ncomp)…", () -> WSInsightCommands.ncomp().showAndRun()),
                item("Edge composition (ecomp)…",         () -> WSInsightCommands.ecomp().showAndRun()),
                item("Triad composition (tcomp)…",        () -> WSInsightCommands.tcomp().showAndRun()),
                item("Cellular microenvironment (cme)…",  () -> WSInsightCommands.cme().showAndRun()),
                new SeparatorMenuItem(),
                item("Export GeoJSON / OME-CSV…", () -> WSInsightCommands.export().showAndRun())
        );
    }

    private static MenuItem item(String label, Runnable r) {
        MenuItem mi = new MenuItem(label);
        mi.setOnAction(ev -> {
            try {
                r.run();
            } catch (Exception ex) {
                LoggerFactory.getLogger(WSInsightExtension.class)
                        .error("WSInsight command failed", ex);
                javafx.scene.control.Alert a =
                        new javafx.scene.control.Alert(javafx.scene.control.Alert.AlertType.ERROR);
                a.setHeaderText("WSInsight command failed");
                a.setContentText(ex.getMessage());
                a.showAndWait();
            }
        });
        return mi;
    }

    @Override public String getName() { return "WSInsight"; }

    @Override public String getDescription() {
        return "QuPath GUI wrapper around the WSInsight Docker image for whole-slide "
                + "patch-level classification, single-cell inference, and graph-based "
                + "spatial analytics.";
    }

    @Override public Version getQuPathVersion() { return Version.parse("0.7.0"); }

    @Override public GitHubRepo getRepository() {
        return GitHubRepo.create(getName(), "huangch", "wsinsight");
    }
}
