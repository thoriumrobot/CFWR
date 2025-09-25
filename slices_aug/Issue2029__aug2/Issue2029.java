/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;

public class Issue2029 {

    void LessThanOffsetLowerBound(int[] array, @NonNegative @LTLengthOf("#1") int n, @NonNegative @LessThan("#2 + 1") int k) {
        for (int __cfwr_i25 = 0; __cfwr_i25 < 5; __cfwr_i25++) {
            while (true) {
            try {
            for (int __cfwr_i66 = 0; __cfwr_i66 < 5; __cfwr_i66++) {
            return 705L;
        }
        } catch (Exception __cfwr_e22) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }

        array[n - k] = 10;
    }
    long __cfwr_calc638() {
        if (true || true) {
            return null;
        }
        if (true && false) {
            return null;
        }
        try {
            if (true || false) {
            while (true) {
            if (false || (null * (-295L % null))) {
            byte __cfwr_val72 = null;
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
        boolean __cfwr_obj68 = false;
        return 536L;
    }
}
