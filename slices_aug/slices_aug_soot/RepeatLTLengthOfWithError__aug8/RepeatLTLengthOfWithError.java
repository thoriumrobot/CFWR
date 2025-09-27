/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.EnsuresLTLengthOf;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOfIf;

public class RepeatLTLengthOfWithError {

    @EnsuresLTLengthOf.List({ @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3"), @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2") })
    @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
    public void withpostconditionfunc1() {
        long __cfwr_elem90 = 370L;

        v1 = value1.length() - 3;
        v2 = value2.length() - 3;
        v3 = value3.length() - 3;
    }
    public static byte __cfwr_helper822(Float __cfwr_p0, char __cfwr_p1, Boolean __cfwr_p2) {
        try {
            for (int __cfwr_i30 = 0; __cfwr_i30 < 4; __cfwr_i30++) {
            Double __cfwr_entry10 = null;
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }
        return null;
    }
}
