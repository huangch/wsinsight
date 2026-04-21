package qupath.ext.wsinsight.runner;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

public class PathMapperTest {

    @Test
    void hostToContainer_mapsSimplePath() {
        PathMapper m = new PathMapper(List.of(
                new PathMapper.Mount(Paths.get("/data/slides"), "/slides")));
        assertEquals("/slides/case1/x.svs", m.hostToContainer("/data/slides/case1/x.svs"));
    }

    @Test
    void hostToContainer_mapsMountRoot() {
        PathMapper m = new PathMapper(List.of(
                new PathMapper.Mount(Paths.get("/data/slides"), "/slides")));
        assertEquals("/slides", m.hostToContainer("/data/slides"));
    }

    @Test
    void hostToContainer_returnsNullWhenNotCovered() {
        PathMapper m = new PathMapper(List.of(
                new PathMapper.Mount(Paths.get("/data/slides"), "/slides")));
        assertNull(m.hostToContainer("/elsewhere/x.svs"));
    }

    @Test
    void hostToContainer_prefersLongerRoot() {
        PathMapper m = new PathMapper(List.of(
                new PathMapper.Mount(Paths.get("/data"), "/mnt/data"),
                new PathMapper.Mount(Paths.get("/data/slides"), "/slides")));
        assertEquals("/slides/x.svs", m.hostToContainer("/data/slides/x.svs"));
        assertEquals("/mnt/data/other/y.svs", m.hostToContainer("/data/other/y.svs"));
    }

    @Test
    void containerToHost_roundTrips() {
        PathMapper m = new PathMapper(List.of(
                new PathMapper.Mount(Paths.get("/data/slides"), "/slides")));
        Path host = m.containerToHost("/slides/case1/x.svs");
        assertEquals(Paths.get("/data/slides/case1/x.svs"), host);
    }
}
