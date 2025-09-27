/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class ArrayAssignmentSameLen {

    void test3(int[] a, @LTLengthOf("#1") int i, @NonNegative int x) {
        Integer __cfwr_elem89 = null;

        int[] c1 = a;
        @LTLengthOf(value = { "c1", "c1" }, offset = { "0", "x" })
        int z = i;
    }
    protected static Boolean __cfwr_util895(long __cfwr_p0, char __cfwr_p1, long __cfwr_p2) {
        while (true) {
            try {
            for (int __cfwr_i82 = 0; __cfwr_i82 < 7; __cfwr_i82++) {
            for (int __cfwr_i94 = 0; __cfwr_i94 < 4; __cfwr_i94++) {
            if (false && (10.40 | null)) {
            return null;
        }
        }
        }
        } catch (Exception __cfwr_e18) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    public byte __cfwr_func564() {
        for (int __cfwr_i21 = 0; __cfwr_i21 < 5; __cfwr_i21++) {
            return null;
        }
        return null;
    }
}
