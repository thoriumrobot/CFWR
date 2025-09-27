/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;

public class Issue2029 {

    void LessThanOffsetLowerBound(int[] array, @NonNegative @LTLengthOf("#1") int n, @NonNegative @LessThan("#2 + 1") int k) {
        for (int __cfwr_i51 = 0; __cfwr_i51 < 1; __cfwr_i51++) {
            try {
            return null;
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        }

        array[n - k] = 10;
    }
    private static Object __cfwr_helper370(double __cfwr_p0) {
        Integer __cfwr_var7 = null;
        if (true && true) {
            for (int __cfwr_i90 = 0; __cfwr_i90 < 9; __cfwr_i90++) {
            for (int __cfwr_i96 = 0; __cfwr_i96 < 9; __cfwr_i96++) {
            try {
            while (true) {
            try {
            for (int __cfwr_i35 = 0; __cfwr_i35 < 1; __cfwr_i35++) {
            try {
            if (false || false) {
            char __cfwr_elem83 = 'd';
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e48) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e33) {
            // ignore
        }
        }
        }
        }
        try {
            while (false) {
            Integer __cfwr_var49 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e4) {
            // ignore
        }
        return null;
    }
}
