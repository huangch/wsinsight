package qupath.ext.wsinsight.commands;

import java.util.List;

/**
 * Static catalogue of every WSInsight CLI subcommand exposed through the
 * extension menu. Each factory builds a {@link GenericCommandDialog} configured
 * with the parameters documented in {@code SKILL.md §4} of the WSInsight
 * repository.
 * <p>
 * Defaults are conservative; users tweak values in the rendered form before
 * launching. Paths that correspond to slides or results are marked for
 * {@link qupath.ext.wsinsight.runner.PathMapper} translation so host-side
 * paths chosen in the browse dialogs are rewritten into container paths.
 */
public final class WSInsightCommands {

    private WSInsightCommands() {}

    public static GenericCommandDialog run() {
        return new GenericCommandDialog("WSInsight — run", "run", List.of(
                ParamSpec.path("--wsi-dir", "WSI directory (host)", true, true,
                        "Directory containing whole-slide images."),
                ParamSpec.path("--results-dir", "Results directory (host)", true, true,
                        "Directory where WSInsight writes outputs."),
                ParamSpec.stringOpt("--model", "Model",
                        "breast-tumor-resnet34.tcga-brca",
                        "WSInfer/WSInsight Model Zoo identifier."),
                ParamSpec.intOpt("--batch-size", "Batch size", "32", ""),
                ParamSpec.intOpt("--num-workers", "Dataloader workers", "4", ""),
                ParamSpec.boolFlag("--force", "Overwrite existing outputs", false, "")
        ));
    }

    public static GenericCommandDialog patch() {
        return new GenericCommandDialog("WSInsight — patch", "patch", List.of(
                ParamSpec.path("--wsi-dir", "WSI directory (host)", true, true, ""),
                ParamSpec.path("--results-dir", "Results directory (host)", true, true, ""),
                ParamSpec.intOpt("--patch-size", "Patch size (px)", "224", ""),
                ParamSpec.intOpt("--patch-spacing", "Patch spacing (µm)", "0", ""),
                ParamSpec.boolFlag("--force", "Overwrite existing patches", false, "")
        ));
    }

    public static GenericCommandDialog infer() {
        return new GenericCommandDialog("WSInsight — infer", "infer", List.of(
                ParamSpec.path("--wsi-dir", "WSI directory (host)", true, true, ""),
                ParamSpec.path("--results-dir", "Results directory (host)", true, true, ""),
                ParamSpec.stringOpt("--model", "Model",
                        "breast-tumor-resnet34.tcga-brca", ""),
                ParamSpec.intOpt("--batch-size", "Batch size", "32", ""),
                ParamSpec.intOpt("--num-workers", "Dataloader workers", "4", ""),
                ParamSpec.boolFlag("--force", "Overwrite existing outputs", false, "")
        ));
    }

    public static GenericCommandDialog reg() {
        return new GenericCommandDialog("WSInsight — reg", "reg", List.of(
                ParamSpec.path("--wsi-dir", "WSI directory (host)", true, true, ""),
                ParamSpec.path("--results-dir", "Results directory (host)", true, true, ""),
                ParamSpec.stringOpt("--region-model", "Region model", "", "")
        ));
    }

    public static GenericCommandDialog hplot() {
        return new GenericCommandDialog("WSInsight — hplot", "hplot", List.of(
                ParamSpec.path("--wsi-dir", "WSI directory (host)", true, true, ""),
                ParamSpec.path("--results-dir", "Results directory (host)", true, true, ""),
                ParamSpec.intOpt("--k", "k-NN neighbours", "6", "")
        ));
    }

    public static GenericCommandDialog hplotFinalize() {
        return new GenericCommandDialog("WSInsight — hplot-finalize", "hplot-finalize", List.of(
                ParamSpec.path("--results-dir", "Results directory (host)", true, true, "")
        ));
    }

    public static GenericCommandDialog ncomp() {
        return new GenericCommandDialog("WSInsight — ncomp", "ncomp", List.of(
                ParamSpec.path("--wsi-dir", "WSI directory (host)", true, true, ""),
                ParamSpec.path("--results-dir", "Results directory (host)", true, true, "")
        ));
    }

    public static GenericCommandDialog ecomp() {
        return new GenericCommandDialog("WSInsight — ecomp", "ecomp", List.of(
                ParamSpec.path("--wsi-dir", "WSI directory (host)", true, true, ""),
                ParamSpec.path("--results-dir", "Results directory (host)", true, true, "")
        ));
    }

    public static GenericCommandDialog tcomp() {
        return new GenericCommandDialog("WSInsight — tcomp", "tcomp", List.of(
                ParamSpec.path("--wsi-dir", "WSI directory (host)", true, true, ""),
                ParamSpec.path("--results-dir", "Results directory (host)", true, true, "")
        ));
    }

    public static GenericCommandDialog cme() {
        return new GenericCommandDialog("WSInsight — cme", "cme", List.of(
                ParamSpec.path("--results-dir", "Results directory (host)", true, true, ""),
                ParamSpec.intOpt("--k", "Neighbourhoods (k)", "10", "")
        ));
    }

    public static GenericCommandDialog export() {
        return new GenericCommandDialog("WSInsight — export", "export", List.of(
                ParamSpec.path("--wsi-dir", "WSI directory (host)", true, true, ""),
                ParamSpec.path("--results-dir", "Results directory (host)", true, true, ""),
                ParamSpec.choice("--format", "Output format", "geojson",
                        List.of("geojson", "omecsv", "both"), "")
        ));
    }
}
