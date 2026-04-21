package qupath.ext.wsinsight;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.StringProperty;
import qupath.lib.gui.prefs.PathPrefs;

/**
 * Singleton holding user preferences for the WSInsight QuPath extension.
 * <p>
 * Preferences are persisted via {@link PathPrefs} so they survive QuPath
 * restarts, and are exposed to each command class through getters.
 */
public class WSInsightSetup {

    private static final WSInsightSetup INSTANCE = new WSInsightSetup();

    private final StringProperty dockerBinaryProp =
            PathPrefs.createPersistentPreference("wsinsightDockerBinary", "docker");
    private final StringProperty dockerImageProp =
            PathPrefs.createPersistentPreference("wsinsightDockerImage", "huangchtw/wsinsight:latest");
    private final StringProperty gpusProp =
            PathPrefs.createPersistentPreference("wsinsightGpus", "all");
    private final StringProperty shmSizeProp =
            PathPrefs.createPersistentPreference("wsinsightShmSize", "32g");

    private final StringProperty hostWsiRootProp =
            PathPrefs.createPersistentPreference("wsinsightHostWsiRoot", "");
    private final StringProperty hostResultsRootProp =
            PathPrefs.createPersistentPreference("wsinsightHostResultsRoot", "");
    private final StringProperty extraMountsProp =
            PathPrefs.createPersistentPreference("wsinsightExtraMounts", "");

    private final StringProperty zooRegistryProp =
            PathPrefs.createPersistentPreference("wsinsightEnvWsinferZooRegistry", "");
    private final StringProperty s3OptionsProp =
            PathPrefs.createPersistentPreference("wsinsightEnvS3Options", "");
    private final StringProperty cacheDirProp =
            PathPrefs.createPersistentPreference("wsinsightCacheDir", "");
    private final StringProperty kerasHomeProp =
            PathPrefs.createPersistentPreference("wsinsightKerasHome", "");

    private final BooleanProperty autoImportResultsProp =
            PathPrefs.createPersistentPreference("wsinsightAutoImportResults", true);

    private WSInsightSetup() {}

    public static WSInsightSetup getInstance() {
        return INSTANCE;
    }

    public StringProperty dockerBinaryProperty() { return dockerBinaryProp; }
    public StringProperty dockerImageProperty() { return dockerImageProp; }
    public StringProperty gpusProperty() { return gpusProp; }
    public StringProperty shmSizeProperty() { return shmSizeProp; }
    public StringProperty hostWsiRootProperty() { return hostWsiRootProp; }
    public StringProperty hostResultsRootProperty() { return hostResultsRootProp; }
    public StringProperty extraMountsProperty() { return extraMountsProp; }
    public StringProperty zooRegistryProperty() { return zooRegistryProp; }
    public StringProperty s3OptionsProperty() { return s3OptionsProp; }
    public StringProperty cacheDirProperty() { return cacheDirProp; }
    public StringProperty kerasHomeProperty() { return kerasHomeProp; }
    public BooleanProperty autoImportResultsProperty() { return autoImportResultsProp; }

    public String getDockerBinary() { return dockerBinaryProp.get(); }
    public String getDockerImage() { return dockerImageProp.get(); }
    public String getGpus() { return gpusProp.get(); }
    public String getShmSize() { return shmSizeProp.get(); }
    public String getHostWsiRoot() { return hostWsiRootProp.get(); }
    public String getHostResultsRoot() { return hostResultsRootProp.get(); }
    public String getExtraMounts() { return extraMountsProp.get(); }
    public String getZooRegistry() { return zooRegistryProp.get(); }
    public String getS3Options() { return s3OptionsProp.get(); }
    public String getCacheDir() { return cacheDirProp.get(); }
    public String getKerasHome() { return kerasHomeProp.get(); }
    public boolean isAutoImportResults() { return autoImportResultsProp.get(); }
}
