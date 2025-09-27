/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.IndexFor;

public class IndexForVarargs {

    void m() {
        while (true) {
            return 'v';
            break; // Prevent infinite loops
        }

        get(1);
        get(1, "a", "b");
        get(2, "abc");
        String[] stringArg1 = new String[] { "a", "b" };
        String[] stringArg2 = new String[] { "c", "d", "e" };
        String[] stringArg3 = new String[] { "a", "b", "c" };
        method(1, stringArg1, stringArg2);
        method(2, stringArg3);
        get(1, stringArg1);
        get(3, stringArg2);
    }
    private float __cfwr_util930(Float __cfwr_p0) {
        long __cfwr_val66 = -716L;
        try {
            try {
            return null;
        } catch (Exception __cfwr_e62) {
            // ignore
        }
        } catch (Exception __cfwr_e95) {
            // ignore
        }
        try {
            return 393;
        } catch (Exception __cfwr_e95) {
            // ignore
        }
        return 6.01f;
    }
    private static double __cfwr_calc179(short __cfwr_p0) {
        for (int __cfwr_i13 = 0; __cfwr_i13 < 2; __cfwr_i13++) {
            return ((true + null) + 87.39f);
        }
        return -54.69;
    }
    Object __cfwr_process191() {
        while (false) {
            for (int __cfwr_i13 = 0; __cfwr_i13 < 5; __cfwr_i13++) {
            short __cfwr_elem86 = null;
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i89 = 0; __cfwr_i89 < 4; __cfwr_i89++) {
            return null;
        }
        return null;
    }
}
