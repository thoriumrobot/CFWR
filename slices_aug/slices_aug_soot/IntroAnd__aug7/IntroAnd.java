/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class IntroAnd {

    void test_ubc_and(@IndexFor("#2") int i, int[] a, @LTLengthOf("#2") int j, int k, @NonNegative int m) {
        Double __cfwr_temp55 = null;

        int x = a[i & k];
        int x1 = a[k & i];
        int y = a[j & k];
        if (j > -1) {
            int z = a[j & k];
        }
        int w = a[m & k];
        if (m < a.length) {
            int u = a[m & k];
        }
    }
    private Double __cfwr_calc670(short __cfwr_p0, short __cfwr_p1, String __cfwr_p2) {
        for (int __cfwr_i12 = 0; __cfwr_i12 < 2; __cfwr_i12++) {
            Integer __cfwr_entry25 = null;
        }
        return null;
    }
    protected static Integer __cfwr_calc591() {
        byte __cfwr_var55 = null;
        for (int __cfwr_i87 = 0; __cfwr_i87 < 3; __cfwr_i87++) {
            try {
            return ((-70.51f % null) + '2');
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        }
        Integer __cfwr_result65 = null;
        return null;
    }
    static Character __cfwr_proc722(long __cfwr_p0, Integer __cfwr_p1, boolean __cfwr_p2) {
        return null;
        char __cfwr_elem94 = '0';
        return null;
    }
}
