/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.EnsuresLTLengthOf;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOfIf;

public class RepeatLTLengthOfWithError {

    @EnsuresLTLengthOf.List({ @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3"), @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2") })
    @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
    public void withpostconditionfunc1() {
        for (int __cfwr_i94 = 0; __cfwr_i94 < 1; __cfwr_i94++) {
            Integer __cfwr_result54 = null;
        }

        v1 = value1.length() - 3;
        v2 = value2.length() - 3;
        v3 = value3.length() - 3;
    }
    protected static byte __cfwr_aux410(char __cfwr_p0, Long __cfwr_p1) {
        short __cfwr_temp32 = ((true & null) * 239);
        for (int __cfwr_i74 = 0; __cfwr_i74 < 8; __cfwr_i74++) {
            while ((null - 485)) {
            if (true && (-22.33f % 'm')) {
            while (true) {
            while (true) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
        try {
            try {
            int __cfwr_var25 = 605;
        } catch (Exception __cfwr_e53) {
            // ignore
        }
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        return null;
    }
    public static Double __cfwr_helper282(Object __cfwr_p0) {
        try {
            try {
            if (true && ((null >> -58.59) % '3')) {
            float __cfwr_item3 = (('F' + 'f') * 540);
        }
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        return null;
        return null;
    }
}
