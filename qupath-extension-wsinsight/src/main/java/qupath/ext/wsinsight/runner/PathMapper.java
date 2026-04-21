package qupath.ext.wsinsight.runner;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Translates host paths to container paths (and back) based on a configured
 * list of bind mounts. Used to rewrite user-supplied arguments before passing
 * them into the WSInsight Docker container.
 */
public class PathMapper {

    /** One bind mount: host directory backed by a container mount point. */
    public static final class Mount {
        public final Path hostRoot;
        public final String containerRoot; // e.g. "/slides"

        public Mount(Path hostRoot, String containerRoot) {
            this.hostRoot = hostRoot.toAbsolutePath().normalize();
            String trimmed = containerRoot.trim();
            if (trimmed.isEmpty() || !trimmed.startsWith("/"))
                throw new IllegalArgumentException("Container path must start with '/': " + containerRoot);
            // strip trailing slash (except root)
            while (trimmed.length() > 1 && trimmed.endsWith("/"))
                trimmed = trimmed.substring(0, trimmed.length() - 1);
            this.containerRoot = trimmed;
        }

        public String dockerVolumeArg() {
            return hostRoot.toString() + ":" + containerRoot;
        }
    }

    private final List<Mount> mounts;

    public PathMapper(List<Mount> mounts) {
        // Sort by longest host path first so nested mounts resolve correctly.
        List<Mount> sorted = new ArrayList<>(mounts);
        sorted.sort((a, b) -> Integer.compare(
                b.hostRoot.toString().length(),
                a.hostRoot.toString().length()));
        this.mounts = Collections.unmodifiableList(sorted);
    }

    public List<Mount> getMounts() {
        return mounts;
    }

    /**
     * Translate a host path into its container path. Returns {@code null} if
     * no mount covers the given path.
     */
    public String hostToContainer(String hostPath) {
        if (hostPath == null || hostPath.isEmpty())
            return null;
        Path p = Paths.get(hostPath).toAbsolutePath().normalize();
        for (Mount m : mounts) {
            if (p.startsWith(m.hostRoot)) {
                Path rel = m.hostRoot.relativize(p);
                String relStr = rel.toString().replace('\\', '/');
                if (relStr.isEmpty())
                    return m.containerRoot;
                return m.containerRoot + "/" + relStr;
            }
        }
        return null;
    }

    /**
     * Translate a container path back to a host path. Returns {@code null}
     * if no mount covers the given path.
     */
    public Path containerToHost(String containerPath) {
        if (containerPath == null || containerPath.isEmpty())
            return null;
        String normalized = containerPath;
        while (normalized.length() > 1 && normalized.endsWith("/"))
            normalized = normalized.substring(0, normalized.length() - 1);
        for (Mount m : mounts) {
            if (normalized.equals(m.containerRoot)) {
                return m.hostRoot;
            }
            String prefix = m.containerRoot + "/";
            if (normalized.startsWith(prefix)) {
                return m.hostRoot.resolve(normalized.substring(prefix.length()));
            }
        }
        return null;
    }
}
