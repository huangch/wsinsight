package qupath.ext.wsinsight.runner;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.ext.wsinsight.WSInsightSetup;

/**
 * Spawns the WSInsight Docker container, streams its output to a
 * {@link ProgressListener}, and supports cancellation via {@code docker kill}.
 * <p>
 * Thread model: {@link #run()} blocks the calling thread until the container
 * exits, so callers should typically invoke it from a background
 * {@code javafx.concurrent.Task}. {@link #cancel()} is safe to call from any
 * thread.
 */
public class DockerRunner {

    private static final Logger logger = LoggerFactory.getLogger(DockerRunner.class);

    private final String dockerBinary;
    private final String image;
    private final String gpus;
    private final String shmSize;
    private final List<PathMapper.Mount> mounts;
    private final Map<String, String> env;
    private final List<String> wsinsightArgs;

    private final Path cidFile;
    private volatile Process process;
    private volatile boolean cancelled;

    private DockerRunner(Builder b) {
        this.dockerBinary = b.dockerBinary;
        this.image = b.image;
        this.gpus = b.gpus;
        this.shmSize = b.shmSize;
        this.mounts = List.copyOf(b.mounts);
        this.env = new LinkedHashMap<>(b.env);
        this.wsinsightArgs = List.copyOf(b.wsinsightArgs);
        try {
            this.cidFile = Files.createTempFile("wsinsight-cid-" + UUID.randomUUID(), ".cid");
            // docker refuses to write the cidfile if it exists; delete it first.
            Files.deleteIfExists(this.cidFile);
        } catch (IOException e) {
            throw new RuntimeException("Unable to allocate cidfile for docker run", e);
        }
    }

    /** Build the {@code docker run ...} command line. */
    List<String> buildCommand() {
        List<String> cmd = new ArrayList<>();
        cmd.add(dockerBinary);
        cmd.add("run");
        cmd.add("--rm");
        cmd.add("--cidfile");
        cmd.add(cidFile.toString());
        if (gpus != null && !gpus.isBlank() && !"none".equalsIgnoreCase(gpus.trim())) {
            cmd.add("--gpus");
            cmd.add(gpus.trim());
        }
        if (shmSize != null && !shmSize.isBlank()) {
            cmd.add("--shm-size=" + shmSize.trim());
        }
        // On Linux / macOS, running as the invoking user avoids root-owned outputs.
        String os = System.getProperty("os.name", "").toLowerCase();
        if (!os.contains("win")) {
            String uid = System.getenv("UID");
            String gid = System.getenv("GID");
            if (uid == null || uid.isBlank()) uid = tryExec("id", "-u");
            if (gid == null || gid.isBlank()) gid = tryExec("id", "-g");
            if (uid != null && gid != null) {
                cmd.add("--user");
                cmd.add(uid + ":" + gid);
            }
        }
        for (PathMapper.Mount m : mounts) {
            cmd.add("-v");
            cmd.add(m.dockerVolumeArg());
        }
        for (Map.Entry<String, String> e : env.entrySet()) {
            if (e.getValue() == null || e.getValue().isEmpty()) continue;
            cmd.add("-e");
            cmd.add(e.getKey() + "=" + e.getValue());
        }
        cmd.add(image);
        cmd.add("wsinsight");
        cmd.addAll(wsinsightArgs);
        return cmd;
    }

    private static String tryExec(String... argv) {
        try {
            Process p = new ProcessBuilder(argv).redirectErrorStream(true).start();
            try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
                String line = r.readLine();
                p.waitFor();
                return line == null ? null : line.trim();
            }
        } catch (IOException | InterruptedException e) {
            return null;
        }
    }

    /** Launch the container and block until it exits. Safe to invoke once. */
    public int run(ProgressListener listener) throws IOException, InterruptedException {
        List<String> cmd = buildCommand();
        logger.info("Launching WSInsight container: {}", String.join(" ", cmd));
        if (listener != null)
            listener.onLogLine("$ " + String.join(" ", cmd));

        ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true);
        this.process = pb.start();

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                logger.info("wsinsight: {}", line);
                if (listener != null)
                    listener.onLogLine(line);
            }
        }

        int exit = process.waitFor();
        if (cancelled)
            exit = 130; // conventional "cancelled by user" code
        try { Files.deleteIfExists(cidFile); } catch (IOException ignored) {}
        if (listener != null)
            listener.onFinished(exit);
        return exit;
    }

    /** Kill the running container (if any) via {@code docker kill}. */
    public void cancel() {
        cancelled = true;
        String cid = readCid();
        if (cid != null && !cid.isBlank()) {
            try {
                new ProcessBuilder(dockerBinary, "kill", cid)
                        .redirectErrorStream(true)
                        .start()
                        .waitFor();
            } catch (IOException | InterruptedException e) {
                logger.warn("Failed to docker kill {}: {}", cid, e.getMessage());
            }
        } else if (process != null) {
            process.destroy();
        }
    }

    private String readCid() {
        try {
            if (Files.exists(cidFile))
                return Files.readString(cidFile, StandardCharsets.UTF_8).trim();
        } catch (IOException ignored) {}
        return null;
    }

    public static Builder builder() { return new Builder(); }

    /** Fluent builder; all setters return {@code this}. */
    public static final class Builder {
        private String dockerBinary = "docker";
        private String image = "huangchtw/wsinsight:latest";
        private String gpus = "all";
        private String shmSize = "32g";
        private final List<PathMapper.Mount> mounts = new ArrayList<>();
        private final Map<String, String> env = new LinkedHashMap<>();
        private final List<String> wsinsightArgs = new ArrayList<>();

        public Builder dockerBinary(String v) { this.dockerBinary = v; return this; }
        public Builder image(String v) { this.image = v; return this; }
        public Builder gpus(String v) { this.gpus = v; return this; }
        public Builder shmSize(String v) { this.shmSize = v; return this; }
        public Builder mount(PathMapper.Mount m) { this.mounts.add(m); return this; }
        public Builder mounts(List<PathMapper.Mount> ms) { this.mounts.addAll(ms); return this; }
        public Builder env(String k, String v) {
            if (v != null && !v.isEmpty()) this.env.put(k, v);
            return this;
        }
        public Builder args(List<String> a) { this.wsinsightArgs.addAll(a); return this; }
        public Builder arg(String a) { this.wsinsightArgs.add(a); return this; }
        public DockerRunner build() { return new DockerRunner(this); }

        /** Pre-populate from {@link WSInsightSetup} (image, gpus, shm, env, mounts). */
        public Builder fromSetup(WSInsightSetup s) {
            dockerBinary(s.getDockerBinary());
            image(s.getDockerImage());
            gpus(s.getGpus());
            shmSize(s.getShmSize());
            if (!s.getHostWsiRoot().isEmpty())
                mount(new PathMapper.Mount(new File(s.getHostWsiRoot()).toPath(), "/slides"));
            if (!s.getHostResultsRoot().isEmpty())
                mount(new PathMapper.Mount(new File(s.getHostResultsRoot()).toPath(), "/results"));
            for (String entry : s.getExtraMounts().split("[,;\\n]")) {
                String e = entry.trim();
                if (e.isEmpty()) continue;
                int idx = e.lastIndexOf(':');
                if (idx <= 0) continue;
                mount(new PathMapper.Mount(new File(e.substring(0, idx)).toPath(), e.substring(idx + 1)));
            }
            env("WSINFER_ZOO_REGISTRY_PATH", s.getZooRegistry());
            env("S3_STORAGE_OPTIONS", s.getS3Options());
            env("WSINSIGHT_REMOTE_CACHE_DIR", s.getCacheDir());
            env("KERAS_HOME", s.getKerasHome());
            return this;
        }
    }
}
