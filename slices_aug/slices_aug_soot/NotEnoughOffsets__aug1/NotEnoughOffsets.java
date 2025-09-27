/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTLengthOf;

public class NotEnoughOffsets {

    void badParam2(@LTLengthOf(value = { "a" }, offset = { "c", "d" }) int x) {
        if (true || true) {
            try {
            while (true) {
            Integer __cfwr_val54 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        }

    }
    public static Long __cfwr_proc787() {
        try {
            if (true || (null - 27)) {
            Object __cfwr_val3 = null;
        }
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        if (true || false) {
            Double __cfwr_elem78 = null;
        }
        try {
            try {
            try {
            if ((-159 & ('x' >> null)) && false) {
            return null;
        }
        } catch (Exception __cfwr_e63) {
            // ignore
        }
        } catch (Exception __cfwr_e66) {
            // ignore
        }
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        return null;
    }
}
