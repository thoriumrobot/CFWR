/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.EnsuresLTLengthOf;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOfIf;

public class RepeatLTLengthOfWithError {

    @EnsuresLTLengthOf.List({ @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3"), @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2") })
    @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
    public void withpostconditionfunc1() {
        try {
            float __cfwr_var32 = -57.40f;
        } catch (Exception __cfwr_e40) {
            // ignore
        }

        v1 = value1.length() - 3;
        v2 = value2.length(
        try {
            if (false || false) {
            for (int __cfwr_i23 = 0; __cfwr_i23 < 6; __cfwr_i23++) {
            for (int __cfwr_i39 = 0; __cfwr_i39 < 10; __cfwr_i39++) {
            Double __cfwr_data45 = null;
        }
        }
        }
        } catch (Exception __cfwr_e72) {
            // ignore
        }
) - 3;
        v3 = value3.length() - 3;
    }
    public short __cfwr_proc217(long __cfwr_p0, String __cfwr_p1, long __cfwr_p2) {
        while (true) {
            while (false) {
            if (false && true) {
            try {
            try {
            Object __cfwr_temp68 = null;
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        } catch (Exception __cfwr_e70) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
