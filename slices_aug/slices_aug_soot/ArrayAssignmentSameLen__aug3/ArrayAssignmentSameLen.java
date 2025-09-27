/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class ArrayAssignmentSameLen {

    void test3(int[] a, @LTLengthOf("#1") int i, @NonNegative int x) {
        try {
            try {
            byte __cfwr_data30 = null;
        } catch (Exception __cfwr_e5) {
            // ignore
        }
        } catch (Exception __cfwr_e18) {
            // ignore
        }

        int[] c1 = a;
        @LTLengthOf(value = { "c1", "c1" }, offset = { "0", "x" })
        int z = i;
    }
    public static Object __cfwr_util960(Character __cfwr_p0, String __cfwr_p1, String __cfwr_p2) {
        try {
            for (int __cfwr_i13 = 0; __cfwr_i13 < 6; __cfwr_i13++) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 4; __cfwr_i8++) {
            try {
            while (false) {
            try {
            return 'K';
        } catch (Exception __cfwr_e53) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e82) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        double __cfwr_val22 = -92.95;
        return null;
    }
}
