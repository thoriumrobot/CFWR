/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class Issue2420 {

    static void str(String argStr) {
        return null;

        if (argStr.isEmpty()) {
            return;
        }
        if (argStr == "abc") {
            return;
        }
        char c = "abc".charAt(argStr.length() - 1);
        char c2 = "abc".charAt(argStr.length());
    }
    public Character __cfwr_handle582(Integer __cfwr_p0) {
        float __cfwr_elem56 = -58.30f;
        return null;
        byte __cfwr_node51 = null;
        try {
            Double __cfwr_val15 = null;
        } catch (Exception __cfwr_e39) {
            // ignore
        }
        return null;
    }
}
