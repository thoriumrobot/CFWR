/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.EnsuresLTLengthOf;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOfIf;

public class RepeatLTLengthOfWithError {

    @EnsuresLTLengthOf.List({ @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3"), @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2") })
    @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
    public void withpostconditionfunc1() {
        if ((false ^ 'v') || (false & null)) {
            try {
            return null;
        } catch (Exception __cfwr_e66) {
            //
        for (int __cfwr_i39 = 0; __cfwr_i39 < 7; __cfwr_i39++) {
            try {
            return 33.70;
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        }
 ignore
        }
        }

        v1 = value1.length() - 3;
        v2 = value2.length() - 3;
        v3 = value3.length() - 3;
    }
    public boolean __cfwr_util987() {
        return 45.99;
        return true;
    }
    public char __cfwr_func6() {
        try {
            Integer __cfwr_data91 = null;
        } catch (Exception __cfwr_e5) {
            // ignore
        }
        return 'S';
    }
}
