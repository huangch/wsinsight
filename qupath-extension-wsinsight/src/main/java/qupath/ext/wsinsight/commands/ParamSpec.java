package qupath.ext.wsinsight.commands;

import java.util.List;
import java.util.Objects;

/**
 * Declarative specification for a single WSInsight CLI parameter, used to
 * auto-render a JavaFX form in {@link GenericCommandDialog}.
 */
public final class ParamSpec {

    public enum Kind { STRING, INT, DOUBLE, BOOL_FLAG, PATH, CHOICE }

    /** CLI option name, e.g. "--wsi-dir". Use {@code null} for positional args. */
    public final String flag;
    public final String label;
    public final String help;
    public final Kind kind;
    public final String defaultValue;
    public final List<String> choices;
    /** If true and {@link #kind} == PATH, the value will be translated via PathMapper before being passed to wsinsight. */
    public final boolean translatePath;
    /** If true, the option is required; dialog refuses to launch when blank. */
    public final boolean required;

    private ParamSpec(Builder b) {
        this.flag = b.flag;
        this.label = Objects.requireNonNull(b.label);
        this.help = b.help == null ? "" : b.help;
        this.kind = Objects.requireNonNull(b.kind);
        this.defaultValue = b.defaultValue == null ? "" : b.defaultValue;
        this.choices = b.choices == null ? List.of() : List.copyOf(b.choices);
        this.translatePath = b.translatePath;
        this.required = b.required;
    }

    public static Builder builder() { return new Builder(); }

    public static ParamSpec stringOpt(String flag, String label, String defaultValue, String help) {
        return builder().flag(flag).label(label).kind(Kind.STRING).defaultValue(defaultValue).help(help).build();
    }

    public static ParamSpec intOpt(String flag, String label, String defaultValue, String help) {
        return builder().flag(flag).label(label).kind(Kind.INT).defaultValue(defaultValue).help(help).build();
    }

    public static ParamSpec boolFlag(String flag, String label, boolean defaultOn, String help) {
        return builder().flag(flag).label(label).kind(Kind.BOOL_FLAG)
                .defaultValue(Boolean.toString(defaultOn)).help(help).build();
    }

    public static ParamSpec path(String flag, String label, boolean translate, boolean required, String help) {
        return builder().flag(flag).label(label).kind(Kind.PATH)
                .translatePath(translate).required(required).help(help).build();
    }

    public static ParamSpec choice(String flag, String label, String defaultValue, List<String> choices, String help) {
        return builder().flag(flag).label(label).kind(Kind.CHOICE)
                .defaultValue(defaultValue).choices(choices).help(help).build();
    }

    public static final class Builder {
        private String flag;
        private String label;
        private String help;
        private Kind kind;
        private String defaultValue;
        private List<String> choices;
        private boolean translatePath;
        private boolean required;

        public Builder flag(String v) { this.flag = v; return this; }
        public Builder label(String v) { this.label = v; return this; }
        public Builder help(String v) { this.help = v; return this; }
        public Builder kind(Kind v) { this.kind = v; return this; }
        public Builder defaultValue(String v) { this.defaultValue = v; return this; }
        public Builder choices(List<String> v) { this.choices = v; return this; }
        public Builder translatePath(boolean v) { this.translatePath = v; return this; }
        public Builder required(boolean v) { this.required = v; return this; }
        public ParamSpec build() { return new ParamSpec(this); }
    }
}
