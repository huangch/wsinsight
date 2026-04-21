package qupath.ext.wsinsight.commands;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Dialog;
import javafx.scene.control.DialogPane;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.control.Tooltip;
import javafx.scene.control.ButtonType;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;

import qupath.ext.wsinsight.WSInsightSetup;
import qupath.ext.wsinsight.runner.DockerRunner;
import qupath.ext.wsinsight.ui.WSInsightProgressDialog;

/**
 * Generic form for a single WSInsight subcommand. Collects user input for a
 * list of {@link ParamSpec}s, translates paths via the active
 * {@link qupath.ext.wsinsight.runner.PathMapper}, and launches the Docker
 * container through {@link WSInsightProgressDialog}.
 */
public class GenericCommandDialog {

    private final String title;
    private final String subcommand;
    private final List<ParamSpec> specs;

    public GenericCommandDialog(String title, String subcommand, List<ParamSpec> specs) {
        this.title = title;
        this.subcommand = subcommand;
        this.specs = specs;
    }

    /** Show the parameter form; on OK, launch the container and block until it finishes. */
    public void showAndRun() {
        Dialog<Map<String, String>> dialog = new Dialog<>();
        dialog.setTitle(title);
        dialog.setHeaderText("WSInsight — " + subcommand);

        DialogPane pane = dialog.getDialogPane();
        pane.getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);

        GridPane grid = new GridPane();
        grid.setHgap(8);
        grid.setVgap(6);
        grid.setPadding(new Insets(8));

        Map<String, Node> inputs = new LinkedHashMap<>();
        int row = 0;
        for (ParamSpec spec : specs) {
            Label label = new Label(spec.label + (spec.required ? " *" : "") + ":");
            if (!spec.help.isBlank()) label.setTooltip(new Tooltip(spec.help));
            Node input = buildInput(spec);
            grid.add(label, 0, row);
            grid.add(input, 1, row);
            inputs.put(keyFor(spec), input);
            row++;
        }
        pane.setContent(grid);

        dialog.setResultConverter(bt -> {
            if (bt != ButtonType.OK) return null;
            Map<String, String> values = new LinkedHashMap<>();
            for (ParamSpec spec : specs) {
                Node n = inputs.get(keyFor(spec));
                values.put(keyFor(spec), readInput(n));
            }
            return values;
        });

        Map<String, String> result = dialog.showAndWait().orElse(null);
        if (result == null) return;

        // Build argv.
        WSInsightSetup setup = WSInsightSetup.getInstance();
        DockerRunner.Builder rb = DockerRunner.builder().fromSetup(setup);
        rb.arg(subcommand);
        qupath.ext.wsinsight.runner.PathMapper pm = buildPathMapper(setup);

        List<ParamSpec> missing = new ArrayList<>();
        for (ParamSpec spec : specs) {
            String val = result.get(keyFor(spec));
            if (spec.required && (val == null || val.isBlank())) {
                missing.add(spec);
                continue;
            }
            if (val == null || val.isBlank()) continue;

            switch (spec.kind) {
                case BOOL_FLAG:
                    if ("true".equalsIgnoreCase(val)) rb.arg(spec.flag);
                    break;
                case PATH:
                    String resolved = val;
                    if (spec.translatePath) {
                        String mapped = pm.hostToContainer(val);
                        if (mapped == null) {
                            throw new IllegalStateException(
                                    "Path '" + val + "' is not covered by any configured Docker bind mount. "
                                    + "Configure the WSI or Results host root in Preferences, "
                                    + "or add it under 'Extra mounts'.");
                        }
                        resolved = mapped;
                    }
                    if (spec.flag != null) rb.arg(spec.flag);
                    rb.arg(resolved);
                    break;
                default:
                    if (spec.flag != null) rb.arg(spec.flag);
                    rb.arg(val);
                    break;
            }
        }

        if (!missing.isEmpty()) {
            StringBuilder sb = new StringBuilder("Missing required: ");
            for (ParamSpec s : missing) sb.append(s.label).append(", ");
            throw new IllegalArgumentException(sb.toString());
        }

        DockerRunner runner = rb.build();
        WSInsightProgressDialog progress = new WSInsightProgressDialog("WSInsight — " + subcommand, runner);
        progress.showAndRun();
    }

    private static String keyFor(ParamSpec s) {
        return s.flag != null ? s.flag : s.label;
    }

    private Node buildInput(ParamSpec spec) {
        switch (spec.kind) {
            case BOOL_FLAG: {
                CheckBox cb = new CheckBox();
                cb.setSelected("true".equalsIgnoreCase(spec.defaultValue));
                return cb;
            }
            case CHOICE: {
                ChoiceBox<String> box = new ChoiceBox<>();
                box.getItems().addAll(spec.choices);
                if (!spec.defaultValue.isEmpty() && spec.choices.contains(spec.defaultValue))
                    box.setValue(spec.defaultValue);
                else if (!spec.choices.isEmpty())
                    box.setValue(spec.choices.get(0));
                return box;
            }
            case PATH: {
                TextField tf = new TextField(spec.defaultValue);
                tf.setPrefColumnCount(40);
                Button browse = new Button("…");
                browse.setOnAction(ev -> {
                    // Heuristic: if label hints "file", use file chooser, else directory.
                    if (spec.label.toLowerCase().contains("file")) {
                        FileChooser fc = new FileChooser();
                        File f = fc.showOpenDialog(null);
                        if (f != null) tf.setText(f.getAbsolutePath());
                    } else {
                        DirectoryChooser dc = new DirectoryChooser();
                        File f = dc.showDialog(null);
                        if (f != null) tf.setText(f.getAbsolutePath());
                    }
                });
                HBox box = new HBox(4, tf, browse);
                return box;
            }
            default: {
                TextField tf = new TextField(spec.defaultValue);
                tf.setPrefColumnCount(40);
                return tf;
            }
        }
    }

    private String readInput(Node n) {
        if (n instanceof CheckBox cb) return Boolean.toString(cb.isSelected());
        if (n instanceof ChoiceBox<?> cb) {
            Object v = cb.getValue();
            return v == null ? "" : v.toString();
        }
        if (n instanceof TextField tf) return tf.getText();
        if (n instanceof HBox box && !box.getChildren().isEmpty() && box.getChildren().get(0) instanceof TextField tf)
            return tf.getText();
        return "";
    }

    private static qupath.ext.wsinsight.runner.PathMapper buildPathMapper(WSInsightSetup setup) {
        List<qupath.ext.wsinsight.runner.PathMapper.Mount> mounts = new ArrayList<>();
        if (!setup.getHostWsiRoot().isEmpty())
            mounts.add(new qupath.ext.wsinsight.runner.PathMapper.Mount(
                    new File(setup.getHostWsiRoot()).toPath(), "/slides"));
        if (!setup.getHostResultsRoot().isEmpty())
            mounts.add(new qupath.ext.wsinsight.runner.PathMapper.Mount(
                    new File(setup.getHostResultsRoot()).toPath(), "/results"));
        for (String entry : setup.getExtraMounts().split("[,;\\n]")) {
            String e = entry.trim();
            if (e.isEmpty()) continue;
            int idx = e.lastIndexOf(':');
            if (idx <= 0) continue;
            mounts.add(new qupath.ext.wsinsight.runner.PathMapper.Mount(
                    new File(e.substring(0, idx)).toPath(), e.substring(idx + 1)));
        }
        return new qupath.ext.wsinsight.runner.PathMapper(mounts);
    }
}
