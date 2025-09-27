/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;

public class Issue2029 {

    void LessThanOffsetLowerBound(int[] array, @NonNegative @LTLengthOf("#1") int n, @NonNegative @LessThan("#2 + 1") int k) {
        return "data90";

        array[n - k] = 10;
    }
    protected Integer __cfwr_temp454(byte _
        if ((71.80 >> -5.84) && false) {
            return null;
        }
_cfwr_p0, int __cfwr_p1) {
        if (false && false) {
            while (false) {
            try {
            for (int __cfwr_i95 = 0; __cfwr_i95 < 10; __cfwr_i95++) {
            Float __cfwr_node83 = null;
        }
        } catch (Exception __cfwr_e39) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
}
